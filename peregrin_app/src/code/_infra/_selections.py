class Metrics:
    """
    Class holding metrics options for the UI.
    """

    Spot = [
        "Time point",
        "X coordinate",
        "Y coordinate",
        "Distance",
        "Direction"
        "Cumulative length",
        "Cumulative displacement",
        "Cumulative confinement ratio",
    ]

    Track = [
        "Track length",
        "Track displacement",
        "Confinement ratio",
        "Track points",
        "Speed mean",
        "Speed median",
        "Speed max",
        "Speed min",
        "Speed std",
        "Direction mean",
        "Direction std",
    ]

    Time = [
        "Track length",
        "Track displacement",
        "Confinement ratio",
        "Speed"
    ]

    LookFor = {
        "select_id": ["track id", "track identifier", "track"],
        "select_t": ["position t", "t position", "time", "time position", "frame", "t"],
        "select_x": ["position x", "x position", "x coordinate", "coordinate x", "x",],
        "select_y": ["position y", "y position", "y coordinate", "coordinate y", "y"],
    }

    Lut = [
        "Track length", 
        "Track displacement", 
        "Confinement ratio",
        "Track points",
        "Speed instantaneous",
        "Speed mean",
        "Speed median",
        "Speed max",
        "Speed min",
        "Speed std",
        "Direction mean",
        "Direction sd",
    ]

    class Units:
        """
        Class holding units options for the UI.
        """

        TimeUnits = {
            "seconds": "s",
            "minutes": "min",
            "hours": "h",
            "days": "d",
        }

        @staticmethod
        def SetUnits(t: str) -> dict:
            return {
                "Track length": "(µm)",
                "Track displacement": "(µm)",
                "Confinement ratio": "",
                "Track points": "",
                "Time point": f"({t})",
                "X coordinate": "(µm)",
                "Y coordinate": "(µm)",
                "Speed mean": f"(µm·{t}⁻¹)",
                "Speed median": f"(µm·{t}⁻¹)",
                "Speed max": f"(µm·{t}⁻¹)",
                "Speed min": f"(µm·{t}⁻¹)",
                "Speed std": f"(µm·{t}⁻¹)",
                "Speed instantaneous": f"(µm·{t}⁻¹)",
                "Direction mean": "",
                "Direction sd": "",
            }
    

    class Thresholding:

        SpotProperties = [
            "Time point",
            "X coordinate",
            "Y coordinate",
        ]

        TrackProperties = [
            "Track length",
            "Track displacement",
            "Confinement ratio",
            "Track points",
            "Speed mean",
            "Speed std",
            "Direction mean",
        ]

        Properties = TrackProperties + SpotProperties




class Styles:
    """
    Class holding color options for the UI.
    """

    ColorMode = [
        "random colors",
        "random greys",
        "differentiate conditions",
        "differentiate replicates",
        "only-one-color",
        "greyscale LUT",
        "reverse grayscale LUT",
        "jet LUT",
        "brg LUT",
        "cool LUT",
        "hot LUT",
        "inferno LUT",
        "plasma LUT",
        "CMR-map LUT",
        "gist-stern LUT",
        "gnuplot LUT",
        "viridis LUT",
        "cividis LUT",
        "rainbow LUT",
        "turbo LUT",
        "nipy-spectral LUT",
        "gist-ncar LUT",
        "twilight LUT",
        "seismic LUT",
    ]

    PaletteQualitativeMatplotlib = [
        "Set1",
        "Set2",
        "Set3",
        "tab10",
        "Accent",
        "Dark2",
        "Pastel1",
        "Pastel2"
    ]

    PaletteQualitativeSeaborn = [
        "deep", 
        "muted", 
        "bright", 
        "pastel", 
        "dark", 
        "colorblind", 
        "husl",
        "hsl"
    ]

    PaletteQualitativeAltair = [
        "category10",
        "tableau10",
        "observable10",
        "set1",
        "set2",
        "set3",
        "pastel1",
        "pastel2",
        "dark2",
        "accent"
    ]

    Color = [
        "red",
        "darkred",
        "firebrick",
        "crimson",
        "indianred",
        "salmon",
        "lightsalmon",
        "tomato",
        "coral",
        "orange",
        "darkorange",
        "gold",
        "khaki",
        "lemonchiffon",
        "peachpuff",
        "wheat",
        "tan",
        "peru",
        "chocolate",
        "sienna",
        "brown",
        "maroon",
        "yellow",
        "yellowgreen",
        "lawngreen",
        "greenyellow",
        "springgreen",
        "lightgreen",
        "palegreen",
        "mediumspringgreen",
        "mediumseagreen",
        "seagreen",
        "forestgreen",
        "green",
        "darkgreen",
        "olive",
        "teal",
        "turquoise",
        "mediumturquoise",
        "paleturquoise",
        "cyan",
        "darkcyan",
        "aqua",
        "deepskyblue",
        "skyblue",
        "lightblue",
        "powderblue",
        "steelblue",
        "dodgerblue",
        "blue",
        "mediumblue",
        "darkblue",
        "navy",
        "royalblue",
        "slateblue",
        "mediumslateblue",
        "slategrey",
        "mediumorchid",
        "mediumpurple",
        "purple",
        "indigo",
        "darkviolet",
        "violet",
        "orchid",
        "plum",
        "magenta",
        "pink",
        "lightcoral",
        "rosybrown",
        "mistyrose",
        "lavender",
        "linen",
        "white",
        "snow",
        "whitesmoke",
        "lightgrey",
        "silver",
        "grey",
        "darkgrey",
        "dimgray",
        "black",
    ]

    Background = [
        "white",
        "light",
        "mid",
        "dark",
        "black"
    ]

    LineStyle = [
        "solid",
        "dashed",
        "dotted",
        "dashdot",
    ]



class Markers:
    """
    Class holding marker options for the UI.
    """

    TrackHeads = {  # TODO - add smiley face markers which will enable a lut map assigning sad faces to "dead cells", so the value spectrum for the lut map will be something like track length or diffusion coefficient [0, max value in the dataset]
        "circle-full": {"symbol": "o", "fill": True},
        "circle-empty": {"symbol": "o", "fill": False},
        "diamond-full": {"symbol": "D", "fill": True},
        "diamond-empty": {"symbol": "D", "fill": False},
        "pentagon-full": {"symbol": "P", "fill": True},
        "pentagon-empty": {"symbol": "P", "fill": False},
        "hexagon-full": {"symbol": "h", "fill": True},
        "hexagon-empty": {"symbol": "h", "fill": False},
        "octagon-full": {"symbol": "p", "fill": True},
        "octagon-empty": {"symbol": "p", "fill": False},
        "star-full": {"symbol": "*", "fill": True},
        "star-empty": {"symbol": "*", "fill": False},
    }

    PlotlyOpen = [
        "circle-open", 
        "square-open", 
        "triangle-open", 
        "star-open", 
        "diamond-open", 
        "pentagon-open",
    ]

    Emoji = [
        "cell",
        "scaled",
        "trains",
        "random",
        "farm",
        "safari",
        "insects",
        "birds",
        "forest",
        "aquarium",
    ]





class Modes:
    """
    Class holding modes for various functions for the UI.
    """

    Thresholding = [
        "Literal",
        "Normalized 0-1",
        "Quantile",
        "Relative to..."
    ]

    FitModel = {
        "Linear":         "(lambda x, a, b: a * x + b, [1, 0])",
        "Quadratic":       "(lambda x, a, b, c: a * x**2 + b * x + c, [1, 1, 0])",
        "Cubic":           "(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, [1, 1, 1, 0])",
        "Logarithmic":     "(lambda x, a, b: a * np.log(x + 1e-9) + b, [1, 0])",
        "Exponential":     "(lambda x, a, b, c: a * np.exp(b * x) + c, [1, 0.1, 0])",
        "Logistic Growth": "(lambda x, L, k, x0: L / (1 + np.exp(-k * (x - x0))), [max(y), 1, np.median(x)]),",
        "Sine Wave":       "(lambda x, A, w, p, c: A * np.sin(w * x + p) + c, [np.std(y), 1, 0, np.mean(y)]),",
        "Gompertz":        "(lambda x, a, b, c: a * np.exp(-b * np.exp(-c * x)), [max(y), 1, 0.1]),",
        "Power Law":       "(lambda x, a, b: a * np.power(x + 1e-9, b), [1, 1])",
    }
    
    Interpolate = [
        "none",
        "basis",
        "basis-open",
        "basis-closed",
        "bundle",
        "cardinal",
        "cardinal-open",
        "cardinal-closed",
        "catmull-rom",
        "linear",
        "linear-closed",
        "monotone",
        "natural",
        "step",
        "step-before",
        "step-after",
    ]
    
    ExtentError = [
        "std", 
        "min-max",
    ]