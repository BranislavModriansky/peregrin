(function() {
    const idsToRemove = ['animated-canvas', 'tiles-grid-container'];
    idsToRemove.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.remove();
        }
    });
})();