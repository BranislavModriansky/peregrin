// background-boxes.js
// Vanilla JS recreation of shadcn.io background boxes for Shiny for Python

(function () {
  const CONTAINER_ID = 'tiles-grid-container';

  // Cleanup existing to prevent duplicates
  const existing = document.getElementById(CONTAINER_ID);
  if (existing) existing.remove();

  // --- Configuration Constants ---
  const ROWS = 80;
  const COLS = 60;
  
  // Animation & Visuals
  const TRANSITION_IN = "background-color 0.0125s ease-out"; // Delay for lighting up
  const TRANSITION_OUT = "background-color 0.5s ease";    // Delay for fading out
  const STROKE_WIDTH = "1.5";
  const ICON_VIEWBOX = "0 0 24 24"; // Changed back to standard 24x24 for better scaling
  
const COLORS = [
    "#7dd3fc", // sky-300 
    "#f9a8d4", // pink-300
    "#86efac", // green-300
    "#efd562", // yellow-300
    "#fca5a5", // red-300
    "#d8b4fe", // purple-300
    "#93c5fd", // blue-300
    "#a5b4fc", // indigo-300
    "#c4b5fd", // violet-300
];

  function getRandomColor() {
    return COLORS[Math.floor(Math.random() * COLORS.length)];
  }

  // Create container
  const container = document.createElement("div");
  container.id = CONTAINER_ID;
  container.className = "background-boxes-wrapper";
  
  // Force fixed positioning to act as background
  Object.assign(container.style, {
    position: 'fixed',
    top: '0',
    left: '0',
    width: '100%',
    height: '100%',
    zIndex: '0', // Sit behind content but visible
    overflow: 'hidden',
    pointerEvents: 'none' // Let clicks pass through to the app
  });

  // Create Grid
  const grid = document.createElement("div");
  grid.className = "bg-boxes-grid";

  // --- Global Hover Logic ---
  // Since the app content covers the background, standard mouseover won't work.
  // We use elementsFromPoint to "pierce" through the UI and find the tile below.
  let lastHoveredCell = null;
  let ticking = false;

  const handleHover = (x, y) => {
    // Get all elements at cursor position
    // Note: elementsFromPoint only returns elements with pointer-events: auto (or default)
    const elements = document.elementsFromPoint(x, y);
    // Find the cell in the stack
    const cell = elements.find(el => el.classList && el.classList.contains('bg-boxes-cell'));

    if (cell) {
      if (cell !== lastHoveredCell) {
        // Clear previous
        if (lastHoveredCell) {
           lastHoveredCell.style.backgroundColor = "transparent";
           lastHoveredCell.style.transition = TRANSITION_OUT;
        }
        // Highlight new
        cell.style.backgroundColor = getRandomColor();
        cell.style.transition = TRANSITION_IN; // Apply the "in" transition for delayed effect
        lastHoveredCell = cell;
      }
    } else {
      // Cursor moved off grid
      if (lastHoveredCell) {
         lastHoveredCell.style.backgroundColor = "transparent";
         lastHoveredCell.style.transition = TRANSITION_OUT;
         lastHoveredCell = null;
      }
    }
  };

  const onMouseMove = (e) => {
    // Self-cleanup: stop listening if the grid is removed (theme switch)
    if (!document.getElementById(CONTAINER_ID)) {
      document.removeEventListener('mousemove', onMouseMove);
      return;
    }

    if (!ticking) {
      window.requestAnimationFrame(() => {
        handleHover(e.clientX, e.clientY);
        ticking = false;
      });
      ticking = true;
    }
  };

  document.addEventListener('mousemove', onMouseMove);
  // --------------------------

  const fragment = document.createDocumentFragment();

  // SVG Icon template to clone (faster than creating new every time)
  const svgTemplate = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svgTemplate.setAttribute("viewBox", ICON_VIEWBOX);
  svgTemplate.setAttribute("fill", "none");
  svgTemplate.setAttribute("stroke", "currentColor");
  svgTemplate.setAttribute("stroke-width", STROKE_WIDTH);
  svgTemplate.classList.add("bg-boxes-plus-icon");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("stroke-linecap", "round");
  path.setAttribute("stroke-linejoin", "round");
  path.setAttribute("d", "M12 6v12m6-6H6");
  svgTemplate.appendChild(path);

  for (let r = 0; r < ROWS; r++) {
    const colDiv = document.createElement("div");
    colDiv.className = "bg-boxes-column";

    for (let c = 0; c < COLS; c++) {
      const cell = document.createElement("div");
      cell.className = "bg-boxes-cell";
      
      // Enable hit-testing for elementsFromPoint so hover works
      // This overrides the container's pointer-events: none
      cell.style.pointerEvents = 'auto';

      // Add + icon every other cell
      if (r % 2 === 0 && c % 2 === 0) {
        cell.appendChild(svgTemplate.cloneNode(true));
      }

      colDiv.appendChild(cell);
    }
    fragment.appendChild(colDiv);
  }

  grid.appendChild(fragment);
  container.appendChild(grid);

  // Add Mask
  const mask = document.createElement("div");
  mask.className = "background-boxes-mask";
  container.appendChild(mask);

  // Insert into body
  document.body.insertBefore(container, document.body.firstChild);
})();
