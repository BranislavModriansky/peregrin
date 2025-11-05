# Docker Deploying Cheatsheat 

**How to build a docker image and push it to DockerHub**

## Navigate

> [!NOTE] 
>
> **Verify whether the desired branch is selected** before building the docker image:
>
> ```bash 
> git branch --show-current 
> ```
> 
> Switch if needed:
>
> ```bash
> git switch <branch-name>
> ```

1. Navigate to the repository

```bash
cd "repository_path"
```

2. To check whether Dockerfile is accessible run

```bash
ls
```

## Build

> [!WARNING]
>
> The container must expose the same network port as the one used by `app.py` script.

3. Build the docker image. 

```bash 
docker build -f Dockerfile -t <username>/peregrin:<version> .
```

## Push

4. Push the created docker image to DockerHub

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