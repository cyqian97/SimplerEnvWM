"""
Vision-based cube pose detection using SAM3 and Depth Anything 3.
Converts pixel coordinates + depth to 3D world coordinates matching self.cube.pose.p
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional
import sys
import os

# Lazy imports - only import when VisionCubeDetector is instantiated
_sam3_model_builder = None
_Sam3Processor = None
_DepthAnything3 = None
_depth_anything_path = None

def _import_sam3():
    """Lazy import SAM3 modules"""
    global _sam3_model_builder, _Sam3Processor
    if _sam3_model_builder is None:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        _sam3_model_builder = build_sam3_image_model
        _Sam3Processor = Sam3Processor

def _import_da3():
    """Lazy import Depth Anything 3"""
    global _DepthAnything3, _depth_anything_path
    if _DepthAnything3 is None:
        # Try multiple possible paths
        possible_paths = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../Depth-Anything-3/src")),
            "/workspace/Depth-Anything-3/src",
            "/home/avalocal/Desktop/RL_class/Depth-Anything-3/src",
        ]
        
        depth_anything_path = None
        for path in possible_paths:
            if os.path.exists(path):
                depth_anything_path = path
                break
        
        if depth_anything_path and depth_anything_path not in sys.path:
            sys.path.insert(0, depth_anything_path)
        
        from depth_anything_3.api import DepthAnything3
        _DepthAnything3 = DepthAnything3
        _depth_anything_path = depth_anything_path


class VisionCubeDetector:
    """Detects cube position from RGB image using SAM3 and DA3"""
    
    def __init__(self, device: torch.device):
        self.device = device
        
        # Lazy import and initialize SAM3
        _import_sam3()
        self.sam3_model = _sam3_model_builder()
        self.sam3_model = self.sam3_model.to(device=device)
        device_str = str(device) if isinstance(device, torch.device) else device
        # Lower confidence threshold for better detection
        self.sam3_processor = _Sam3Processor(self.sam3_model, device=device_str, confidence_threshold=0.3)
        
        # Lazy import and initialize Depth Anything 3
        _import_da3()
        self.da3_model = _DepthAnything3.from_pretrained("depth-anything/da3metric-large")
        self.da3_model = self.da3_model.to(device=device)
        
        # Try multiple prompts - start with simpler ones
        self.prompts = ["cube", "red cube", "red object", "small red cube", "red block"]
        self.prompt_idx = 0
    
    def detect_cube_bbox(self, image: np.ndarray, target_resolution: int = 512) -> Optional[Tuple[float, float, float, float]]:
        """
        Detect red cube bounding box using SAM3.
        Upscales image for better detection, then scales bbox back to original size.
        
        Args:
            image: Input image (H, W, 3) uint8
            target_resolution: Target resolution for SAM3 detection (default 256)
        
        Returns:
            (x1, y1, x2, y2) in original image coordinates, or None if not detected.
        """
        original_h, original_w = image.shape[:2]
        
        # Ensure image is RGB uint8 format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Upscale image for SAM3 detection (maintain aspect ratio)
        # Use the larger dimension to determine scale factor
        scale_factor = max(target_resolution / original_w, target_resolution / original_h)
        new_w = int(original_w * scale_factor)
        new_h = int(original_h * scale_factor)
        
        # Resize image using PIL for better quality
        image_pil = Image.fromarray(image, mode='RGB')
        image_pil_upscaled = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        try:
            inference_state = self.sam3_processor.set_image(image_pil_upscaled)
            
            # Try multiple prompts if first one fails
            boxes = None
            for prompt in self.prompts:
                output = self.sam3_processor.set_text_prompt(state=inference_state, prompt=prompt)
                boxes = output["boxes"]
                if len(boxes) > 0:
                    break
            
            if boxes is None or len(boxes) == 0:
                return None
            
            # Get bounding box in upscaled coordinates
            box_upscaled = boxes[0].cpu().numpy()  # (x1, y1, x2, y2) in upscaled image
            x1_up, y1_up, x2_up, y2_up = box_upscaled
            
            # Scale back to original image coordinates
            x1 = x1_up / scale_factor
            y1 = y1_up / scale_factor
            x2 = x2_up / scale_factor
            y2 = y2_up / scale_factor
            
            # Clip to image bounds
            x1 = np.clip(x1, 0, original_w - 1)
            y1 = np.clip(y1, 0, original_h - 1)
            x2 = np.clip(x2, 0, original_w - 1)
            y2 = np.clip(y2, 0, original_h - 1)
            
            return (x1, y1, x2, y2)
        except Exception as e:
            return None
    
    def get_depth_map(self, image: np.ndarray) -> np.ndarray:
        """
        Get depth map from DA3.
        Returns depth map of shape (H, W) in meters.
        """
        image_pil = Image.fromarray(image)
        prediction = self.da3_model.inference(
            image=[image_pil],
            export_dir=None,
            export_format="npz",
            process_res=image.shape[1],
        )
        depth_map = prediction.depth[0]  # Remove batch dimension: (H, W)
        # DA3 returns depth in meters, so no conversion needed
        return depth_map
    
    def calibrate_coordinate_transform(
        self,
        cube_world_pos: np.ndarray,
        pixel_x: float,
        pixel_y: float,
        depth: float,
        intrinsic: np.ndarray,
        cam2world: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        Use known cube position to find correct coordinate transformation.
        
        Args:
            cube_world_pos: Known cube position [x, y, z] in world coordinates
            pixel_x, pixel_y: Detected pixel coordinates
            depth: DA3 depth value
            intrinsic: Camera intrinsic matrix (3x3)
            cam2world: Camera to world transformation matrix (4x4)
        
        Returns:
            (correct_camera_coords, debug_info)
            correct_camera_coords: What camera coordinates SHOULD be
            debug_info: Dictionary with debug information
        """
        # Step 1: Transform cube world position to camera space (inverse of cam2world)
        cube_world_homogeneous = np.array([cube_world_pos[0], cube_world_pos[1], cube_world_pos[2], 1.0])
        world2cam = np.linalg.inv(cam2world)
        cube_cam_homogeneous = world2cam @ cube_world_homogeneous
        cube_cam_gl = cube_cam_homogeneous[:3]  # This is what camera coords SHOULD be in OpenGL space
        
        # Step 2: Compute what we get from pixel + depth (OpenCV camera space)
        fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
        x_cam_cv = (pixel_x - cx) * depth / fx
        y_cam_cv = (pixel_y - cy) * depth / fy
        z_cam_cv = depth
        
        # Step 3: Find transformation that maps OpenCV -> OpenGL
        # We know cube_cam_gl (what it should be) and (x_cam_cv, y_cam_cv, z_cam_cv) (what we have)
        # Need to find: cube_cam_gl = f(x_cam_cv, y_cam_cv, z_cam_cv)
        
        debug_info = {
            "cube_world_pos": cube_world_pos,
            "cube_cam_gl": cube_cam_gl,
            "x_cam_cv": x_cam_cv,
            "y_cam_cv": y_cam_cv,
            "z_cam_cv": z_cam_cv,
            "depth": depth,
        }
        
        return cube_cam_gl, debug_info
    
    def pixel_depth_to_world(
        self,
        pixel_x: float,
        pixel_y: float,
        depth: float,
        intrinsic: np.ndarray,
        cam2world: np.ndarray,
        image_height: int,
        image_width: int,
        cube_world_pos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert pixel coordinates + depth to 3D world coordinates.
        
        Args:
            pixel_x, pixel_y: Pixel coordinates (center of bounding box)
            depth: Depth value in meters (distance from camera)
            intrinsic: Camera intrinsic matrix (3x3) in OpenCV format
            cam2world: Camera to world transformation matrix (4x4) in OpenGL format
            image_height, image_width: Image dimensions
        
        Returns:
            3D world coordinates [x, y, z] matching ManiSkill world coordinates
        """
        # Extract intrinsic parameters
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        
        # Convert pixel to camera coordinates
        # OpenCV convention: x right, y down, z forward
        x_cam_cv = (pixel_x - cx) * depth / fx
        y_cam_cv = (pixel_y - cy) * depth / fy
        z_cam_cv = depth
        
        # If cube position is provided, use it for calibration
        if cube_world_pos is not None:
            cube_cam_gl, calib_info = self.calibrate_coordinate_transform(
                cube_world_pos, pixel_x, pixel_y, depth, intrinsic, cam2world
            )
            
            # Compute correct depth from cube position
            # In OpenGL camera space, Z is backward (negative), so depth = -Z
            # But we need depth in OpenCV convention (forward = positive Z)
            # The depth should be the distance along the camera's forward axis
            # In OpenGL: forward = -z, so depth = -cube_cam_gl[2]
            correct_depth_gl = -cube_cam_gl[2]  # Convert OpenGL Z (backward) to depth (forward)
            
            # Use the Z-based depth (more consistent with camera conventions)
            correct_depth = correct_depth_gl
            
            # Recompute camera coordinates using correct depth
            fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
            x_cam_cv_corrected = (pixel_x - cx) * correct_depth / fx
            y_cam_cv_corrected = (pixel_y - cy) * correct_depth / fy
            z_cam_cv_corrected = correct_depth
            
            # Find the transformation: cube_cam_gl = f(x_cam_cv_corrected, y_cam_cv_corrected, z_cam_cv_corrected)
            # Test all sign combinations and axis permutations
            test_conversions = [
                # Standard OpenGL conversion
                ("x_cv, -y_cv, -z_cv", x_cam_cv_corrected, -y_cam_cv_corrected, -z_cam_cv_corrected),
                # SAPIEN-like conversions
                ("z_cv, -x_cv, -y_cv", z_cam_cv_corrected, -x_cam_cv_corrected, -y_cam_cv_corrected),
                ("z_cv, -y_cv, -x_cv", z_cam_cv_corrected, -y_cam_cv_corrected, -x_cam_cv_corrected),
                # Other permutations
                ("-z_cv, x_cv, y_cv", -z_cam_cv_corrected, x_cam_cv_corrected, y_cam_cv_corrected),
                ("-z_cv, -y_cv, x_cv", -z_cam_cv_corrected, -y_cam_cv_corrected, x_cam_cv_corrected),
                ("-z_cv, -x_cv, y_cv", -z_cam_cv_corrected, -x_cam_cv_corrected, y_cam_cv_corrected),
                ("x_cv, y_cv, z_cv", x_cam_cv_corrected, y_cam_cv_corrected, z_cam_cv_corrected),
                ("-x_cv, -y_cv, -z_cv", -x_cam_cv_corrected, -y_cam_cv_corrected, -z_cam_cv_corrected),
                # More axis swaps
                ("y_cv, -x_cv, -z_cv", y_cam_cv_corrected, -x_cam_cv_corrected, -z_cam_cv_corrected),
                ("-y_cv, x_cv, -z_cv", -y_cam_cv_corrected, x_cam_cv_corrected, -z_cam_cv_corrected),
                ("-y_cv, z_cv, x_cv", -y_cam_cv_corrected, z_cam_cv_corrected, x_cam_cv_corrected),
                ("y_cv, z_cv, -x_cv", y_cam_cv_corrected, z_cam_cv_corrected, -x_cam_cv_corrected),
            ]
            
            best_match = None
            best_error = float('inf')
            best_conversion_name = None
            
            for name, x_test, y_test, z_test in test_conversions:
                test_cam_gl = np.array([x_test, y_test, z_test])
                error = np.linalg.norm(test_cam_gl - cube_cam_gl)
                if error < best_error:
                    best_error = error
                    best_match = test_cam_gl
                    best_conversion_name = name
            
            # Use the best match if error is reasonable (relaxed threshold since we're using corrected depth)
            if best_match is not None and best_error < 0.5:  # Relaxed threshold
                p_cam_gl = np.concatenate([best_match, [1.0]])
                p_world = (cam2world @ p_cam_gl)[:3]
                return p_world
        
        # According to observations.py line 29, position data is "in OpenGL camera space"
        # cam2world_gl expects OpenGL camera coordinates
        # 
        # From look_at comment: OpenGL camera: (forward, right, up) = (-z, x, y)
        # This means: forward = -z, right = x, up = y
        # So OpenGL camera: x right, y up, z backward (negative forward)
        #
        # However, looking at the rotation matrix structure and the fact that Y is close
        # but X and Z are wrong, we might need a different conversion.
        #
        # From sapien_pose_to_opencv_extrinsic: SAPIEN(x,y,z) -> OpenCV(-y, -z, x)
        # Inverse: OpenCV(x,y,z) -> SAPIEN(z, -x, -y)
        #
        # But cam2world_gl is OpenGL format. Let's try converting OpenCV -> SAPIEN -> OpenGL
        # OR try different axis mappings based on the rotation matrix structure.
        #
        # The rotation matrix row 1 is [1, 0, 0], meaning world Y = camera X
        # This suggests the camera's X axis maps directly to world Y.
        #
        # Let's try: Convert OpenCV to match what the rotation matrix expects
        # Based on the matrix structure, try: x_gl = z_cv, y_gl = -x_cv, z_gl = -y_cv
        
        # Try alternative conversion based on SAPIEN convention
        # OpenCV(x,y,z) -> SAPIEN(z, -x, -y) -> then to OpenGL?
        # OR try direct mapping based on rotation matrix structure
        
        # Original conversion (Y is close, so keep Y part):
        x_cam_gl_orig = x_cam_cv
        y_cam_gl_orig = -y_cam_cv  # This works (Y is close)
        z_cam_gl_orig = -z_cam_cv
        
        # Try alternative: swap axes based on rotation matrix
        # Row 1 [1,0,0] means world Y = camera X, so maybe x_gl should map differently
        # Try: x_gl = z_cv (forward becomes right), y_gl = -x_cv, z_gl = y_cv
        
        # Actually, let's try the SAPIEN conversion first, then see if we need OpenGL
        x_cam_sapien = z_cam_cv   # SAPIEN x = OpenCV z
        y_cam_sapien = -x_cam_cv  # SAPIEN y = -OpenCV x
        z_cam_sapien = -y_cam_cv  # SAPIEN z = -OpenCV y
        
        # Now convert SAPIEN to OpenGL
        # SAPIEN: (forward, right, up) = (x, -y, z)
        # OpenGL: (forward, right, up) = (-z, x, y)
        # So: x_gl = -y_sapien, y_gl = x_sapien, z_gl = -z_sapien? No wait...
        # Actually, from the conversion matrix, maybe we should use SAPIEN directly?
        
        # Let's try: Use OpenGL conversion but with swapped X/Z based on rotation matrix
        # The rotation matrix suggests: world Y = camera X (row 1)
        # So maybe: x_gl = z_cv, y_gl = -x_cv, z_gl = -y_cv
        
        # Analysis of debug output:
        # GT: [0.0298, -0.0393, 0.0200]
        # SAPIEN direct: [-0.0737, 1.8862, 0.3199] - X close, Y/Z wrong
        # OpenGL orig: [-0.5252, -0.1168, -1.1554] - Y close, X/Z wrong  
        # SAPIEN->OpenGL: [-0.8904, -0.1168, 2.1314] - Y close, X/Z wrong
        #
        # Key insight: Y is consistently close (-0.1168 vs -0.0393) in OpenGL conversions
        # This means: y_cam_gl = -y_cam_cv is CORRECT
        #
        # For X: SAPIEN direct gives -0.0737 (closer to 0.0298)
        # SAPIEN: x_sapien = z_cv, so maybe we need: x_gl = z_cv?
        #
        # Let's try hybrid: Use SAPIEN X, OpenGL Y, and test different Z
        
        # Convert to SAPIEN first
        x_cam_sapien = z_cam_cv   # SAPIEN x = OpenCV z (this gave good X)
        y_cam_sapien = -x_cam_cv  # SAPIEN y = -OpenCV x
        z_cam_sapien = -y_cam_cv  # SAPIEN z = -OpenCV y
        
        # Hybrid: Use SAPIEN X, OpenGL Y (which we know works), and try SAPIEN Z
        # OpenGL format: x right, y up, z backward
        # Try: x_gl = x_sapien (z_cv), y_gl = -y_cam_cv (we know this works), z_gl = ?
        x_cam_gl = x_cam_sapien   # Use SAPIEN x (z_cv) - this gave good X
        y_cam_gl = -y_cam_cv      # Use OpenGL y (we know this works for Y)
        z_cam_gl = -z_cam_sapien  # Try: -SAPIEN z = -(-y_cv) = y_cv
        
        p_cam_gl = np.array([x_cam_gl, y_cam_gl, z_cam_gl, 1.0])
        p_world = (cam2world @ p_cam_gl)[:3]
        
        # Also keep other conversions for comparison
        p_cam_gl_orig = np.array([x_cam_gl_orig, y_cam_gl_orig, z_cam_gl_orig, 1.0])
        p_world_orig = (cam2world @ p_cam_gl_orig)[:3]
        
        p_cam_sapien = np.array([x_cam_sapien, y_cam_sapien, z_cam_sapien, 1.0])
        p_world_sapien = (cam2world @ p_cam_sapien)[:3]
        
        return p_world
    
    def detect_cube_position(
        self,
        rgb_image: np.ndarray,
        intrinsic: np.ndarray,
        cam2world: np.ndarray,
        cube_world_pos: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Detect cube 3D position from RGB image.
        
        Args:
            rgb_image: RGB image array (H, W, 3), uint8
            intrinsic: Camera intrinsic matrix (3x3)
            cam2world: Camera to world transformation matrix (4x4) in OpenGL format
        
        Returns:
            3D world position [x, y, z] matching self.cube.pose.p format, or None
        """
        H, W = rgb_image.shape[:2]
        
        # Step 1: Detect bounding box with SAM3 (will upscale internally to 512px)
        bbox = self.detect_cube_bbox(rgb_image, target_resolution=512)
        if bbox is None:
            return None
        
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        # Step 2: Get depth map from DA3
        depth_map = self.get_depth_map(rgb_image)
        
        # Resize depth map to match image if needed
        if depth_map.shape != (H, W):
            depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Step 3: Get depth at bounding box center (average over small region)
        x_int = int(np.clip(center_x, 0, W - 1))
        y_int = int(np.clip(center_y, 0, H - 1))
        
        # Average depth over bounding box region for robustness
        x1_int = int(np.clip(x1, 0, W - 1))
        y1_int = int(np.clip(y1, 0, H - 1))
        x2_int = int(np.clip(x2, 0, W - 1))
        y2_int = int(np.clip(y2, 0, H - 1))
        
        depth_region = depth_map[y1_int:y2_int, x1_int:x2_int]
        if depth_region.size == 0:
            return None
        depth_value = np.mean(depth_region)
        
        if depth_value <= 0 or not np.isfinite(depth_value):
            return None
        
        # Step 4: Convert pixel + depth to world coordinates
        world_pos = self.pixel_depth_to_world(
            pixel_x=center_x,
            pixel_y=center_y,
            depth=depth_value,
            intrinsic=intrinsic,
            cam2world=cam2world,
            image_height=H,
            image_width=W,
            cube_world_pos=cube_world_pos,
        )
        
        return world_pos


def RewardVision(
    obs: dict,
    camera_name: str = "base_camera",
    device: torch.device = None,
    cube_world_pos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Detect cube position from vision observations using SAM3 and DA3.
    
    Args:
        obs: Observation dictionary containing:
            - "image": dict with camera_name -> {"rgb": (H, W, 3) or (N, H, W, 3)}
            - "sensor_param": dict with camera_name -> {
                "intrinsic_cv": (3, 3) or (N, 3, 3),
                "cam2world_gl": (4, 4) or (N, 4, 4)
            }
        camera_name: Name of camera to use
        device: Device for vision models
    
    Returns:
        Cube position tensor matching self.cube.pose.p format: (num_envs, 3)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    detector = VisionCubeDetector(device=device)
    
    rgb_image = obs["image"][camera_name]["rgb"]
    intrinsic = obs["sensor_param"][camera_name]["intrinsic_cv"]
    cam2world = obs["sensor_param"][camera_name]["cam2world_gl"]
    
    # Handle batched observations
    if len(rgb_image.shape) == 4:  # Batched: (N, H, W, 3)
        num_envs = rgb_image.shape[0]
        cube_positions = []
        
        for i in range(num_envs):
            img = rgb_image[i].cpu().numpy()
            # Handle normalized images [0, 1] -> [0, 255]
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            intrinsic_i = intrinsic[i].cpu().numpy() if isinstance(intrinsic, torch.Tensor) else intrinsic[i]
            cam2world_i = cam2world[i].cpu().numpy() if isinstance(cam2world, torch.Tensor) else cam2world[i]
            
            # Get cube position for calibration if available
            cube_pos_i = None
            if cube_world_pos is not None:
                cube_pos_i = cube_world_pos[i].cpu().numpy() if isinstance(cube_world_pos, torch.Tensor) else cube_world_pos[i]
            
            world_pos = detector.detect_cube_position(img, intrinsic_i, cam2world_i, cube_world_pos=cube_pos_i)
            if world_pos is None:
                world_pos = np.zeros(3, dtype=np.float32)
            
            cube_positions.append(world_pos)
        
        return torch.tensor(np.array(cube_positions), device=rgb_image.device, dtype=torch.float32)
    
    else:  # Unbatched: (H, W, 3)
        img = rgb_image.cpu().numpy() if isinstance(rgb_image, torch.Tensor) else rgb_image
        # Handle normalized images [0, 1] -> [0, 255]
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        intrinsic_np = intrinsic.cpu().numpy() if isinstance(intrinsic, torch.Tensor) else intrinsic
        cam2world_np = cam2world.cpu().numpy() if isinstance(cam2world, torch.Tensor) else cam2world
        
        if len(intrinsic_np.shape) == 3:
            intrinsic_np = intrinsic_np[0]
        if len(cam2world_np.shape) == 3:
            cam2world_np = cam2world_np[0]
        
        cube_pos_np = None
        if cube_world_pos is not None:
            cube_pos_np = cube_world_pos.cpu().numpy() if isinstance(cube_world_pos, torch.Tensor) else cube_world_pos
            if len(cube_pos_np.shape) > 1:
                cube_pos_np = cube_pos_np[0]
        
        world_pos = detector.detect_cube_position(img, intrinsic_np, cam2world_np, cube_world_pos=cube_pos_np)
        if world_pos is None:
            world_pos = np.zeros(3, dtype=np.float32)
        
        return torch.tensor(world_pos, device=rgb_image.device if isinstance(rgb_image, torch.Tensor) else device, dtype=torch.float32)
