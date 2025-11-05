# Peregrin ~ A Py-Shiny application for exploratory analysis of cell tracking data

![Screenshot](https://github.com/BranislavModriansky/peregrin/blob/re-structure/_exeplary/Screenshot.png?raw=true)

<!-- TODO: A multiframe video compiling various screen recordings of different features -->

## About

Application designed for scientists, allowing **exploratory analysis of cell tracking data via an interactive, browser-based UI** built with [Py-Shiny](https://shiny.posit.co/py/) and containerized with [Docker](https://www.docker.com/) for reproducible deployment.

### Key Features and Flows

#### Data Input

- Import raw tabular - CSV (recommended) / XLSX / XLS - data files containing information about spots of individual trajectories;
- Imported data must contain *track identificator*, *time point*, *x coordinate* and *y coordinate* columns;
- Optionally, raw input may also contain other information;
- Run Peregrin and compute track statistics;
- Display computed data and download them in CSV format.

#### Filtering

- Availible thresholding methods include *Literal*, *Normalized 0-1*, *Percentile*, and *Relative to (a selected value)*.

#### Visualization

- Generate interactive and highly customizable plots covering:
    - Track reconstruction and animation;
    - Time lag statistics visualizations;
    - Time charts;
    - Superplots.
- Export figures in SVG format and more.

<!-- TODO: Describe the problem this project solves, the motivation, and the goals. -->
<!-- TODO: List key features and a brief overview of the architecture/tech stack if helpful. -->
<!-- TODO: Current status/roadmap (optional). -->

## How to Install and Launch

- Use **Docker** for **quick set-up** and **easy configuration**
- Alternatively, use a source code editor (e.g. VSCode) for development, code modification and editing or debugging options. 

### Running the App Using Docker

> [!NOTE]
>
> [Docker desktop](https://docs.docker.com/desktop/) must be installed and running to set-up Peregrin docker image!

#### Set-up

Create a folder into which the docker image will be pulled. Navigate to the created folder:

```bash
cd <your_folder>
```

Pull the docker image:

```bash
docker pull branislavmodriansky/peregrin:<version>
```

> **See the list of all accessible images**
>
> ```bash
> docker image ls
> ```

After succesfully pulling the `peregrin:<version>` docker image, it should be ready to be launched.

#### Initialization

To run the app:
- Open docker desktop -> containers and start the *image*
- Alternatively, the app can be launched from the image directory via:

```bash
docker run branislavmodriansky/peregrin:<version>
```

or

```bash
docker run -p <port>:<port> branislavmodriansky/peregrin:<version> shiny run --host 0.0.0.0 --port <port>
```

<!-- Callout options: [!IMPORTANT], [!NOTE], [!TIP], [!WARNING], [!CAUTION] -->

<!-- ### Running the App In VSCode


## User's guide

## Examples -->

