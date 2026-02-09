# Peregrin ~ A Py-Shiny application for exploratory analysis of cell tracking data


<!-- 

TODO: A multiframe gif "video" compiling various screen recordings of different features 


embed as:
![GIF](https://github.com/BranislavModriansky/peregrin/blob/re-structure/media/data_display.gif)


-->

<!-- ## Contents:

1. [About](#about)
    1. [Key Features and Flows](#key-features-and-flows)
        1. [Data Input](#data-input)
        2. [Filtering](#filtering)
        3. [Visualization](#visualization)
    2. [Documentation](#documentation)
3. [How to Install and Launch](#how-to-install-and-launch)
    1. [Running the App Using Docker](#running-the-app-using-docker)
        1. [Set-up](#set-up)
        2. [Initialization](#initialization) -->

> [!IMPORTANT]
>
> This project is currently under active development. Many more features are planned to be implemented. Availible features may be incomplete or undatble. The goal of this project is to create a sophisticated, open source, open access tool (web app + python library of the source code) with a wide range of functionalities like data computations, filtering (1D and 2D), highest quality data visualizations. Designed by and for scientists.
> If you are interested in working with Peregrin, I will be very happy to help you as well as listen to your feedback, so please contact me:
> email: 548323@mail.muni.cz


## About

Peregrin is a an application designed for researchers studying cell migration. It allows **exploratory analysis of cell tracking data via an interactive, browser-based UI** built with [Py-Shiny](https://shiny.posit.co/py/) and containerized with [Docker](https://www.docker.com/) for reproducible deployment.

### Key Features and Flows

#### Import Data

- Import: CSV / XML / XLSX / XLS - data files containing information about spots of individual trajectories;
- Imported data must contain *track identificator*, *time point*, *x coordinate* and *y coordinate* columns;
- Optionally, input may also contain other information;

#### Run

- Run Peregrin and compute track statistics;
- Display computed data and download them in a CSV format.

#### Apply Filters

- 1D filtering options include: *Literal*, *Normalized 0-1*, *Percentile*, *Relative to (a selected value)*.

#### Visualize

- Generate highly customizable figures, covering:
    - Track reconstruction and animation
    - Polar histograms
    - Richard Martin Edler von Mises distribution lineplots and heatmaps
    - Mean Squared Displacement charts
    - Mean turning angles per frame lags colormesh
    - <del> Time series </del> 🏗️
    - <del> Superplots </del> 🏗️

- Export figures in SVG format.





## Documentation

> [!TIP]
>
> See the [Current State - Road Map](https://github.com/BranislavModriansky/peregrin/wiki/Current-State-%E2%80%90-Road-Map)

<!--

> 
> See the [User's Guide](https://github.com/BranislavModriansky/peregrin/wiki/User's-Guide) page at Peregrin Wiki for a **walk-through of the app's features**!
> 
> See the [Under the Hood: Data Processing, Computation & Visualization](https://github.com/BranislavModriansky/peregrin/wiki/Under-the-Hood:-Data-Processing,-Computation-&-Visualization)

-->


<!-- TODO: Describe the problem this project solves, the motivation, and the goals. -->
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


