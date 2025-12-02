// TODO: Let the user turn off this script while keeping Console-1 the theme

(function() {
    // Cleanup existing canvas if present to prevent duplicates
    const existingCanvas = document.getElementById('animated-canvas');
    if (existingCanvas) {
        existingCanvas.remove();
    }

    // Create and setup canvas
    const canvas = document.createElement('canvas');
    canvas.id = 'animated-canvas'; // Add ID for easy removal
    const ctx = canvas.getContext('2d');
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.zIndex = '0'; // Sits just above body background, below content
    canvas.style.pointerEvents = 'none'; // Let clicks pass through
    document.body.insertBefore(canvas, document.body.firstChild);

    // --- Configuration Object ---
    // Expose config to window so it can be modified from outside (e.g. Python/Shiny)
    window.PeregrinGridConfig = window.PeregrinGridConfig || {
        // Control
        ENABLED: true,
        // Grid & Physics
        GRID_SPACING: 12.5,
        DOT_RADIUS: 0.5,
        INFLUENCE_RADIUS: 750,
        MOUSE_FORCE: 0.75,
        SPRING_STIFFNESS: 0.035,
        DAMPING: 0.75,
        MOUSE_LAG: 0.75,
        // Visuals
        COLOR_R: 130,
        COLOR_G: 218,
        COLOR_B: 240,
        BASE_ALPHA: 0.35,
        // Glow Logic
        GLOW_INTENSITY: 5.25,
        GLOW_POWER: 3,          // <- updated from UI
        GLOW_DISPLACEMENT_NORM: 50,
        ALPHA_SMOOTHING: 0.5,
        // Tangle/Aggregation Prevention
        TANGLE_THRESHOLD: 10,   // <- updated from UI
        FADE_RADIUS_MULT: 2
    };
    const cfg = window.PeregrinGridConfig;

    // --- Global updater called from Shiny-injected <script> ---
    // Example call (emitted by Python):
    // window.PeregrinGridUpdateConfig({MOUSE_FORCE: 2.0, ...});
    window.PeregrinGridUpdateConfig = function(message) {
        if (!message) return;
        for (const key in message) {
            if (Object.prototype.hasOwnProperty.call(cfg, key)) {
                cfg[key] = message[key];
            }
        }
        if (Object.prototype.hasOwnProperty.call(message, "GRID_SPACING")) {
            initDots();
        }
    };

    let width, height;
    let mouseX = -1000;
    let mouseY = -1000;
    // Lagging mouse position for smoother force application
    let lagMouseX = -1000;
    let lagMouseY = -1000;
    let dots = [];

    function initDots() {
        dots = [];
        // Extend grid generation beyond screen edges to prevent gaps during pull
        const extension = 400;
        for (let x = -extension; x < width + extension; x += cfg.GRID_SPACING) {
            for (let y = -extension; y < height + extension; y += cfg.GRID_SPACING) {
                dots.push({
                    originX: x,
                    originY: y,
                    x: x,
                    y: y,
                    vx: 0,
                    vy: 0,
                    currentAlpha: cfg.BASE_ALPHA // Track alpha for smoothing
                });
            }
        }
    }

    function resize() {
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = width;
        canvas.height = height;
        initDots();
    }

    window.addEventListener('resize', resize);
    resize();

    // Named handler for cleanup
    const mouseMoveHandler = (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
    };
    document.addEventListener('mousemove', mouseMoveHandler);

    let debugLastLog = 0;

    function animate(timestamp) {
        // Stop animation loop and cleanup listeners if canvas is removed
        if (!document.body.contains(canvas)) {
            window.removeEventListener('resize', resize);
            document.removeEventListener('mousemove', mouseMoveHandler);
            return;
        }

        // Debug: log current cfg once per second
        if (timestamp !== undefined && timestamp - debugLastLog > 1000) {
            console.log("[proton_grid] animate tick, cfg:", cfg);
            debugLastLog = timestamp;
        }

        // If disabled, just clear and schedule next frame
        if (!cfg.ENABLED) {
            ctx.clearRect(0, 0, width, height);
            requestAnimationFrame(animate);
            return;
        }

        ctx.clearRect(0, 0, width, height);
        
        // Update lag mouse position (Smooths the force movement)
        if (lagMouseX === -1000) {
            lagMouseX = mouseX;
            lagMouseY = mouseY;
        } else {
            lagMouseX += (mouseX - lagMouseX) * cfg.MOUSE_LAG;
            lagMouseY += (mouseY - lagMouseY) * cfg.MOUSE_LAG;
        }

        // Set base color once
        ctx.fillStyle = `rgb(${cfg.COLOR_R}, ${cfg.COLOR_G}, ${cfg.COLOR_B})`;

        for (let i = 0; i < dots.length; i++) {
            const dot = dots[i];

            // Use real mouse for physics (Instant pull)
            const dx = mouseX - dot.x;
            const dy = mouseY - dot.y;
            const distSq = dx * dx + dy * dy;
            
            // Mouse interaction (Attraction)
            if (distSq < cfg.INFLUENCE_RADIUS * cfg.INFLUENCE_RADIUS && distSq > 1) {
                const dist = Math.sqrt(distSq);
                const force = (cfg.INFLUENCE_RADIUS - dist) / cfg.INFLUENCE_RADIUS;
                
                // Tangle Avoidance
                let avoidanceFactor = 1;
                if (dist < cfg.TANGLE_THRESHOLD) {
                    avoidanceFactor = dist / cfg.TANGLE_THRESHOLD;
                }

                // Physics: Pull towards mouse
                const effectiveForce = force * avoidanceFactor * cfg.MOUSE_FORCE;
                
                dot.vx += (dx / dist) * effectiveForce;
                dot.vy += (dy / dist) * effectiveForce;
            }

            // Spring force (return to origin)
            const springDx = dot.originX - dot.x;
            const springDy = dot.originY - dot.y;
            
            dot.vx += springDx * cfg.SPRING_STIFFNESS;
            dot.vy += springDy * cfg.SPRING_STIFFNESS;

            // Apply physics
            dot.vx *= cfg.DAMPING;
            dot.vy *= cfg.DAMPING;
            
            dot.x += dot.vx;
            dot.y += dot.vy;

            // --- Glow Calculation ---
            
            // 1. Calculate displacement
            const displacementSq = (dot.x - dot.originX) ** 2 + (dot.y - dot.originY) ** 2;
            const displacement = Math.sqrt(displacementSq);
            
            // 2. Calculate Target Glow (Power curve)
            let glow = Math.pow(displacement / cfg.GLOW_DISPLACEMENT_NORM, cfg.GLOW_POWER) * cfg.GLOW_INTENSITY;

            // 3. Fade out near the center of force (where dots aggregate)
            // Use lagMouse for the fade out logic (Lagged suppression)
            const lagDx = lagMouseX - dot.x;
            const lagDy = lagMouseY - dot.y;
            const lagDistSq = lagDx * lagDx + lagDy * lagDy;

            const fadeRadius = cfg.TANGLE_THRESHOLD * cfg.FADE_RADIUS_MULT;
            if (lagDistSq < fadeRadius * fadeRadius) {
                const dist = Math.sqrt(lagDistSq);
                const fade = dist / fadeRadius;
                glow *= fade;
            }
            
            let targetAlpha = cfg.BASE_ALPHA + glow;
            if (targetAlpha > 1.0) targetAlpha = 1.0;

            // 4. Smooth Alpha Transition (Fixes flashing)
            dot.currentAlpha += (targetAlpha - dot.currentAlpha) * cfg.ALPHA_SMOOTHING;

            // Draw dot
            ctx.globalAlpha = dot.currentAlpha;
            ctx.beginPath();
            ctx.arc(dot.x, dot.y, cfg.DOT_RADIUS, 0, Math.PI * 2);
            ctx.fill();
        }
        requestAnimationFrame(animate);
    }

    animate();
})();
