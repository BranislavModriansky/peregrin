thresholding = {
    "literal": "Literal",
    "percentile": "Percentile",
}

colors = [
    'black',
    'blue',
    'brown',
    'chocolate',
    'coral',
    'crimson',
    'cyan',
    'darkblue',
    'darkcyan',
    'darkgreen',
    'darkmagenta',
    'darkorange',
    'darkred',
    'darkviolet',
    'deepskyblue',
    'dodgerblue',
    'firebrick',
    'forestgreen',
    'gold',
    'gray',
    'green',
    'grey',
    'indianred',
    'indigo',
    'khaki',
    'lavender',
    'lawngreen',
    'lemonchiffon',
    'lightblue',
    'lightcoral',
    'lightgray',
    'lightgreen',
    'lightgrey',
    'lightsalmon',
    'linen',
    'magenta',
    'maroon',
    'mediumblue',
    'mediumorchid',
    'mediumpurple',
    'mediumseagreen',
    'mediumslateblue',
    'mediumspringgreen',
    'mediumturquoise',
    'mediumvioletred',
    'mistyrose',
    'navy',
    'olive',
    'orange',
    'orchid',
    'palegreen',
    'paleturquoise',
    'palevioletred',
    'peachpuff',
    'peru',
    'pink',
    'plum',
    'powderblue',
    'purple',
    'red',
    'rosybrown',
    'royalblue',
    'salmon',
    'sandybrown',
    'seagreen',
    'sienna',
    'silver',
    'skyblue',
    'slateblue',
    'slategray',
    'slategrey',
    'snow',
    'springgreen',
    'steelblue',
    'tan',
    'teal',
    'tomato',
    'turquoise',
    'violet',
    'wheat',
    'whitesmoke',
    'yellow',
    'yellowgreen'
    ]



sns_palletes = [
    'deep', 
    'muted', 
    'bright', 
    'pastel', 
    'dark', 
    'colorblind', 
    'Set1', 
    'Set2', 
    'Set3', 
    'tab10', 
    'tab20', 
    'tab20c'
    ]

color_modes = [
    'random colors',
    'random greys',
    'differentiate conditions/replicates',
    'only-one-color',
    'greyscale LUT', 
    'jet LUT', 
    'brg LUT', 
    'hot LUT', 
    'gnuplot LUT', 
    'viridis LUT', 
    'rainbow LUT', 
    'turbo LUT', 
    'nipy-spectral LUT', 
    'gist-ncar LUT'
    ]


cmaps_quantitative = [
    'greyscale', 
    'jet', 
    'brg', 
    'hot', 
    'gnuplot', 
    'viridis', 
    'rainbow', 
    'turbo', 
    'nipy-spectral', 
    'gist-ncar'
    ]

cmaps_qualitative = [
    'Set1',
    'Set2',
    'Set3',
    'tab10',
    'Accent',
    'Dark2',
    'Paired',
    'Pastel1',
    'Pastel2',
    ]

interpolation = [
    'None',
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
    "step-after"
    ]

extent = [
    'std', 
    'min-max' 
    ]

background = [
    'light', 
    'dark'
    ]


models = {
        "(AA) Linear":         "(lambda x, a, b: a * x + b, [1, 0])",
        "(A) Quadratic":       "(lambda x, a, b, c: a * x**2 + b * x + c, [1, 1, 0])",
        "(B) Cubic":           "(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, [1, 1, 1, 0])",
        "(C) Logarithmic":     "(lambda x, a, b: a * np.log(x + 1e-9) + b, [1, 0])",
        "(D) Exponential":     "(lambda x, a, b, c: a * np.exp(b * x) + c, [1, 0.1, 0])",
        "(E) Logistic Growth": "(lambda x, L, k, x0: L / (1 + np.exp(-k * (x - x0))), [max(y), 1, np.median(x)]),",
        "(F) Sine Wave":       "(lambda x, A, w, p, c: A * np.sin(w * x + p) + c, [np.std(y), 1, 0, np.mean(y)]),",
        "(G) Gompertz":        "(lambda x, a, b, c: a * np.exp(-b * np.exp(-c * x)), [max(y), 1, 0.1]),",
        "(H) Power Law":       "(lambda x, a, b: a * np.power(x + 1e-9, b), [1, 1])"
    }