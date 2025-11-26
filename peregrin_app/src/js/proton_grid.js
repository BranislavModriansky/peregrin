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

    let width, height;
    let mouseX = -1000;
    let mouseY = -1000;
    // Lagging mouse position for smoother force application
    let lagMouseX = -1000;
    let lagMouseY = -1000;
    let dots = [];

    // --- Configuration Constants ---
    
    // Grid & Physics
    const GRID_SPACING = 12.5;      
    const DOT_RADIUS = 0.5;       
    const INFLUENCE_RADIUS = 750; 
    const MOUSE_FORCE = 0.75;     
    const SPRING_STIFFNESS = 0.035; 
    const DAMPING = 0.75;
    const MOUSE_LAG = 0.75; // 0.0 to 1.0 - Lower is more delayed/smoother force

    // Visuals
    const COLOR_R = 130;
    const COLOR_G = 218;
    const COLOR_B = 240;
    const BASE_ALPHA = 0.35;     
    
    // Glow Logic
    const GLOW_INTENSITY = 5.25; 
    const GLOW_POWER = 3; // Curve of the glow (higher = more rapid rise)
    const GLOW_DISPLACEMENT_NORM = 50; // Distance at which glow normalizes
    const ALPHA_SMOOTHING = 0.5; // 0.0 to 1.0 - Lower is smoother transitions (fixes flashing)
    
    // Tangle/Aggregation Prevention
    const TANGLE_THRESHOLD = 10; 
    const FADE_RADIUS_MULT = 2; // Multiplier for the dark center radius

    function initDots() {
        dots = [];
        // Extend grid generation beyond screen edges to prevent gaps during pull
        const extension = 400;
        for (let x = -extension; x < width + extension; x += GRID_SPACING) {
            for (let y = -extension; y < height + extension; y += GRID_SPACING) {
                dots.push({
                    originX: x,
                    originY: y,
                    x: x,
                    y: y,
                    vx: 0,
                    vy: 0,
                    currentAlpha: BASE_ALPHA // Track alpha for smoothing
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

    function animate() {
        // Stop animation loop and cleanup listeners if canvas is removed
        if (!document.body.contains(canvas)) {
            window.removeEventListener('resize', resize);
            document.removeEventListener('mousemove', mouseMoveHandler);
            return;
        }

        ctx.clearRect(0, 0, width, height);
        
        // Update lag mouse position (Smooths the force movement)
        if (lagMouseX === -1000) {
            lagMouseX = mouseX;
            lagMouseY = mouseY;
        } else {
            lagMouseX += (mouseX - lagMouseX) * MOUSE_LAG;
            lagMouseY += (mouseY - lagMouseY) * MOUSE_LAG;
        }

        // Set base color once
        ctx.fillStyle = `rgb(${COLOR_R}, ${COLOR_G}, ${COLOR_B})`;

        for (let i = 0; i < dots.length; i++) {
            const dot = dots[i];

            // Use real mouse for physics (Instant pull)
            const dx = mouseX - dot.x;
            const dy = mouseY - dot.y;
            const distSq = dx * dx + dy * dy;
            
            // Mouse interaction (Attraction)
            if (distSq < INFLUENCE_RADIUS * INFLUENCE_RADIUS && distSq > 1) {
                const dist = Math.sqrt(distSq);
                const force = (INFLUENCE_RADIUS - dist) / INFLUENCE_RADIUS;
                
                // Tangle Avoidance
                let avoidanceFactor = 1;
                if (dist < TANGLE_THRESHOLD) {
                    avoidanceFactor = dist / TANGLE_THRESHOLD;
                }

                // Physics: Pull towards mouse
                const effectiveForce = force * avoidanceFactor * MOUSE_FORCE;
                
                dot.vx += (dx / dist) * effectiveForce;
                dot.vy += (dy / dist) * effectiveForce;
            }

            // Spring force (return to origin)
            const springDx = dot.originX - dot.x;
            const springDy = dot.originY - dot.y;
            
            dot.vx += springDx * SPRING_STIFFNESS;
            dot.vy += springDy * SPRING_STIFFNESS;

            // Apply physics
            dot.vx *= DAMPING;
            dot.vy *= DAMPING;
            
            dot.x += dot.vx;
            dot.y += dot.vy;

            // --- Glow Calculation ---
            
            // 1. Calculate displacement
            const displacementSq = (dot.x - dot.originX) ** 2 + (dot.y - dot.originY) ** 2;
            const displacement = Math.sqrt(displacementSq);
            
            // 2. Calculate Target Glow (Power curve)
            let glow = Math.pow(displacement / GLOW_DISPLACEMENT_NORM, GLOW_POWER) * GLOW_INTENSITY;

            // 3. Fade out near the center of force (where dots aggregate)
            // Use lagMouse for the fade out logic (Lagged suppression)
            const lagDx = lagMouseX - dot.x;
            const lagDy = lagMouseY - dot.y;
            const lagDistSq = lagDx * lagDx + lagDy * lagDy;

            const fadeRadius = TANGLE_THRESHOLD * FADE_RADIUS_MULT;
            if (lagDistSq < fadeRadius * fadeRadius) {
                const dist = Math.sqrt(lagDistSq);
                const fade = dist / fadeRadius;
                glow *= fade;
            }
            
            let targetAlpha = BASE_ALPHA + glow;
            if (targetAlpha > 1.0) targetAlpha = 1.0;

            // 4. Smooth Alpha Transition (Fixes flashing)
            dot.currentAlpha += (targetAlpha - dot.currentAlpha) * ALPHA_SMOOTHING;

            // Draw dot
            ctx.globalAlpha = dot.currentAlpha;
            ctx.beginPath();
            ctx.arc(dot.x, dot.y, DOT_RADIUS, 0, Math.PI * 2);
            ctx.fill();
        }
        requestAnimationFrame(animate);
    }

    animate();
})();
