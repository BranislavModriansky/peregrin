# Peregrin ~ A Py-Shiny application for exploratory analysis of cell tracking data


<!-- 

TODO: A multiframe gif "video" compiling various screen recordings of different features 


embed as:
![GIF](https://github.com/BranislavModriansky/peregrin/blob/re-structure/media/data_display.gif)


-->

## About

Peregrin is a an application designed for researchers studying cell migration. It allows **exploratory analysis of cell tracking data via an interactive, browser-based UI** built with [Py-Shiny](https://shiny.posit.co/py/) and containerized with [Docker](https://www.docker.com/) for reproducible deployment.


> [!TIP]
> 
> See the [User's Guide](https://github.com/BranislavModriansky/peregrin/wiki/Peregrin#users-guide) for a walk-through of the app's features!
> 
> See a more detailed documentation of what's [Under the Hood: Computation & Visualization](https://github.com/BranislavModriansky/peregrin/wiki/Peregrin#under-the-hood:-computation-&-visualization)


### Key Features and Flows

#### Data Input

- Import - CSV (recommended) / XLSX / XLS - tabular data files containing information about spots of individual trajectories;
- Imported data must contain *track identificator*, *time point*, *x coordinate* and *y coordinate* columns;
- Optionally, input may also contain other information;
- Run Peregrin and compute track statistics;
- Display computed data and download them in a CSV format.

#### Filtering

- Availible (1D) thresholding methods include: *Literal*, *Normalized 0-1*, *Percentile*, *Relative to (a selected value)*.

#### Visualization

- Generate interactive and highly customizable plots, covering:
    - Track reconstruction and animation;
    - Time lag statistics;
    - Time series;
    - Superplots.
- Export figures in SVG format and more.

<!-- TODO: Describe the problem this project solves, the motivation, and the goals. -->
<!-- TODO: List key features and a brief overview of the architecture/tech stack if helpful. -->
<!-- TODO: Current status/roadmap (optional). -->

## How to Install and Launch

- Use **Docker** for **quick set-up** and **easy configuration**

### Running the App Using Docker

> [!NOTE]
>
> [Docker desktop](https://docs.docker.com/desktop/) must be installed and running to set-up Peregrin.

#### Set-up

1. Create a folder inside of which you would like to store the docker image.
2. Navigate to the folder:

```bash
cd <your_folder>
```

3. Pull the docker image:

```bash
docker pull branislavmodriansky/peregrin:<version>
```

> *See the list of all accessible images*
>
> ```bash
> docker image ls
> ```

After succesfully pulling the `peregrin:<version>` docker image, it should be ready to be launched.


#### Initialization

4. Run the app: <br>
&nbsp; a) Open Docker Desktop -> Containers and start the container. <br>
&nbsp; b) Alternatively, launch from the image directory:

try:

```bash
docker run branislavmodriansky/peregrin:<version>
```

or

```bash
docker run -p <port>:<port> branislavmodriansky/peregrin:<version> shiny run --host 0.0.0.0 --port <port>
```

<!-- Callout options: [!IMPORTANT], [!NOTE], [!TIP], [!WARNING], [!CAUTION] -->


