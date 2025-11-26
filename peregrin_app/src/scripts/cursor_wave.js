(function() {
    // Create and setup canvas
    const canvas = document.createElement('canvas');
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

    // Configuration
    const GRID_SPACING = 40;
    const DOT_RADIUS = 1.5;
    const INFLUENCE_RADIUS = 200; // Size of the "spacetime" distortion
    const MAX_DISPLACEMENT = 50;  // How much dots are pushed away
    const DOT_COLOR = 'rgba(84, 110, 122, 0.3)'; // Subtle muted blue-grey

    function resize() {
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = width;
        canvas.height = height;
    }

    window.addEventListener('resize', resize);
    resize();

    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });

    function animate() {
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = DOT_COLOR;

        // Loop through grid points
        for (let x = 0; x < width + GRID_SPACING; x += GRID_SPACING) {
            for (let y = 0; y < height + GRID_SPACING; y += GRID_SPACING) {
                
                // Calculate distance to mouse
                const dx = x - mouseX;
                const dy = y - mouseY;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                let drawX = x;
                let drawY = y;

                // Apply spacetime distortion (repulsion)
                if (distance < INFLUENCE_RADIUS) {
                    const force = (INFLUENCE_RADIUS - distance) / INFLUENCE_RADIUS;
                    // Cubic easing for smoother "gravity well" feel
                    const ease = Math.pow(force, 3); 
                    
                    const angle = Math.atan2(dy, dx);
                    const moveDistance = ease * MAX_DISPLACEMENT;
                    
                    drawX += Math.cos(angle) * moveDistance;
                    drawY += Math.sin(angle) * moveDistance;
                }

                // Draw dot
                ctx.beginPath();
                ctx.arc(drawX, drawY, DOT_RADIUS, 0, Math.PI * 2);
                ctx.fill();
            }
        }
        requestAnimationFrame(animate);
    }

    animate();
})();
