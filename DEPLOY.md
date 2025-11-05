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

## References
> * **[Docker's Python guide](https://docs.docker.com/language/python/)**
> * **[Getting started](https://docs.docker.com/go/get-started-sharing/)**