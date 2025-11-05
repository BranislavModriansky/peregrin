# Docker Cheatsheat ~ Building a docker image and pushing it to DockerHub

> [!IMPORTANT] 
> **Make sure that the desired branch is selected** before building the docker image!
> ```bash 
> git branch --show-current 
> ```
> Switch if needed:
> ```bash
> git switch <branch-name>
> ```

## Build

1. Navigate to the repository

```bash
cd "repository_path"
```

2. To check whether Dockerfile is accessible run

```bash
ls
```
> [!IMPORTANT] 
> Make sure that the container will be exposed at the correct network port, corresponding to the one listed when running the `app.py` script

3. Build the docker image. 

```bash 
docker build -f Dockerfile -t <username>/peregrin:<version> .
```

Push the created docker image to DockerHub

```bash
docker push <username>/peregrin:<version>
```







### Deploying your application to the cloud

First, build your image, e.g.: `docker build -t myapp .`.
If your cloud uses a different CPU architecture than your development
machine (e.g., you are on a Mac M1 and your cloud provider is amd64),
you'll want to build the image for that platform, e.g.:
`docker build --platform=linux/amd64 -t myapp .`.

Then, push it to your registry, e.g. `docker push myregistry.com/myapp`.

Consult Docker's [getting started](https://docs.docker.com/go/get-started-sharing/)
docs for more detail on building and pushing.

### References
* [Docker's Python guide](https://docs.docker.com/language/python/)