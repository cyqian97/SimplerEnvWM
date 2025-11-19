## Use in docker
Since this repo requires vulkan, and vulkan installation is tricky. I used the Dockerfile from [Nvidia k8s-samples](https://github.com/NVIDIA/k8s-samples/tree/main/deployments/container/vulkan) to build a base image and then applied my own Dockerfile. The image is then started using the docker-compose.yml file.
 The inference was successful in the docker image.