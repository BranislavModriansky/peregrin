(function () {
    const BUTTON_ID = "customize_theme";
    const PANEL_ID = "theme_customization_panel";
    const HEADER_ID = "theme_customization_panel_header";
    const BODY_ID = "theme_customization_panel_body";
    const STYLE_ID = "theme_customization_panel_styles";
    const CONTROLS_MOUNT_ID = "theme_customization_controls_mount";

    const OPEN_DURATION = 380;
    const CLOSE_DURATION = 240;
    const GAP = 10;
    const EDGE_PADDING = 8;

    let initialized = false;
    let dragState = null;
    let userMovedPanel = false;
    let activeAnimation = null;
    let controlsObserver = null;

    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    function getButton() {
        return document.getElementById(BUTTON_ID);
    }

    function getPanel() {
        return document.getElementById(PANEL_ID);
    }

    function getControlsMount() {
        return document.getElementById(CONTROLS_MOUNT_ID);
    }

    function cancelActiveAnimation() {
        if (!activeAnimation) return;
        activeAnimation.cancel();
        activeAnimation = null;
    }

    function injectStyles() {
        if (document.getElementById(STYLE_ID)) return;

        const style = document.createElement("style");
        style.id = STYLE_ID;
        style.textContent = `
            #${PANEL_ID} {
                position: fixed;
                left: 0;
                top: 0;
                width: min(460px, calc(100vw - 24px));
                max-height: min(76vh, 760px);
                display: none;
                overflow: hidden;
                z-index: 2147483000;
                border-radius: 18px;
                border: 1px solid color-mix(in srgb, var(--bs-border-color, rgba(0, 0, 0, 0.15)) 72%, white 28%);
                background:
                    linear-gradient(
                        180deg,
                        color-mix(in srgb, var(--bs-body-bg, #ffffff) 92%, white 8%) 0%,
                        var(--bs-body-bg, #ffffff) 100%
                    );
                color: var(--bs-body-color, #212529);
                box-shadow:
                    0 28px 80px rgba(0, 0, 0, 0.26),
                    0 10px 30px rgba(0, 0, 0, 0.16),
                    inset 0 1px 0 rgba(255, 255, 255, 0.16);
                backdrop-filter: blur(12px) saturate(1.05);
                -webkit-backdrop-filter: blur(12px) saturate(1.05);
                transform-origin: top right;
                will-change: transform, opacity, left, top, border-radius;
            }

            #${PANEL_ID}.is-open {
                display: block;
            }

            #${HEADER_ID} {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.75rem;
                padding: 0.95rem 1rem;
                border-bottom: 1px solid var(--bs-border-color-translucent, rgba(0, 0, 0, 0.12));
                background:
                    linear-gradient(
                        180deg,
                        color-mix(in srgb, var(--bs-tertiary-bg, #f8f9fa) 85%, white 15%) 0%,
                        var(--bs-tertiary-bg, #f8f9fa) 100%
                    );
                cursor: grab;
                user-select: none;
            }

            #${HEADER_ID}.dragging {
                cursor: grabbing;
            }

            #${HEADER_ID} .theme-panel-title {
                margin: 0;
                font-size: 1rem;
                font-weight: 600;
                letter-spacing: 0.01em;
            }

            #${PANEL_ID} .theme-panel-close {
                border: 0;
                background: transparent;
                color: inherit;
                width: 2rem;
                height: 2rem;
                border-radius: 0.6rem;
                font-size: 1.25rem;
                line-height: 1;
                cursor: pointer;
            }

            #${PANEL_ID} .theme-panel-close:hover {
                background: rgba(0, 0, 0, 0.08);
            }

            #${BODY_ID} {
                padding: 1rem;
                overflow: auto;
                max-height: calc(min(76vh, 760px) - 60px);
            }

            #${BODY_ID} .theme-controls-wrap {
                display: block;
            }

            #${BODY_ID} .theme-controls-header {
                margin-bottom: 0.85rem;
            }

            #${BODY_ID} .theme-controls-heading {
                font-size: 0.95rem;
                font-weight: 700;
            }

            #${BODY_ID} .theme-controls-subheading {
                opacity: 0.72;
                font-size: 0.8rem;
                margin-top: 0.15rem;
            }

            #${BODY_ID} .theme-control-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.75rem;
            }

            #${BODY_ID} .theme-color-control {
                padding: 0.8rem;
                border-radius: 14px;
                border: 1px solid var(--bs-border-color-translucent, rgba(0, 0, 0, 0.12));
                background: color-mix(in srgb, var(--bs-body-bg, #ffffff) 90%, white 10%);
            }

            #${BODY_ID} .theme-color-label {
                display: block;
                margin-bottom: 0.45rem;
                font-size: 0.82rem;
                font-weight: 600;
            }

            #${BODY_ID} .theme-color-row {
                display: flex;
                align-items: center;
                gap: 0.65rem;
            }

            #${BODY_ID} .theme-color-input {
                flex: 0 0 52px;
                min-width: 52px;
                cursor: pointer;
            }

            #${BODY_ID} .theme-color-value {
                margin: 0;
                font-size: 0.78rem;
                opacity: 0.8;
                user-select: text;
            }

            #${BODY_ID} .theme-panel-placeholder {
                color: var(--bs-secondary-color, #6c757d);
            }

            @media (max-width: 640px) {
                #${PANEL_ID} {
                    width: min(96vw, 460px);
                }

                #${BODY_ID} .theme-control-grid {
                    grid-template-columns: 1fr;
                }
            }
        `;
        document.head.appendChild(style);
    }

    function ensureControlsVisible() {
        const mount = getControlsMount();
        if (!mount) return;
        mount.style.display = "block";
    }

    function attachControls(panel) {
        const mount = getControlsMount();
        const body = panel ? panel.querySelector(`#${BODY_ID}`) : null;

        if (!mount || !body) return;

        mount.style.display = "block";

        if (mount.parentElement !== body) {
            body.replaceChildren(mount);
        }
    }

    function startControlsObserver() {
        if (controlsObserver) return;

        const mount = getControlsMount();
        if (!mount) return;

        controlsObserver = new MutationObserver(function () {
            const panel = getPanel();
            if (!panel) return;
            attachControls(panel);
        });

        controlsObserver.observe(mount, {
            childList: true,
            subtree: true
        });
    }

    function createPanel() {
        let panel = getPanel();
        if (panel) return panel;

        panel = document.createElement("div");
        panel.id = PANEL_ID;
        panel.className = "card shadow";
        panel.setAttribute("aria-hidden", "true");
        panel.dataset.open = "false";

        panel.innerHTML = `
            <div id="${HEADER_ID}" class="card-header">
                <div class="theme-panel-title">Customize theme</div>
                <button type="button" class="theme-panel-close" aria-label="Close">&times;</button>
            </div>
            <div id="${BODY_ID}" class="card-body">
                <div class="theme-panel-placeholder">Loading theme controls...</div>
            </div>
        `;

        document.body.appendChild(panel);

        const closeButton = panel.querySelector(".theme-panel-close");
        closeButton.addEventListener("click", function (event) {
            event.preventDefault();
            event.stopPropagation();
            closePanel();
        });

        enableDragging(panel);
        attachControls(panel);

        return panel;
    }

    function showPanel(panel) {
        panel.classList.add("is-open");
        panel.style.display = "";
        panel.style.visibility = "";
        panel.style.pointerEvents = "auto";
    }

    function hidePanel(panel) {
        panel.classList.remove("is-open");
        panel.style.display = "";
        panel.style.visibility = "";
        panel.style.pointerEvents = "";
        panel.style.opacity = "";
        panel.style.transform = "";
        panel.style.borderRadius = "";
        panel.style.filter = "";
    }

    function placePanelFromButton(panel, button) {
        const buttonRect = button.getBoundingClientRect();

        showPanel(panel);
        panel.style.visibility = "hidden";
        panel.style.left = "0px";
        panel.style.top = "0px";

        const panelRect = panel.getBoundingClientRect();

        let left = buttonRect.right - panelRect.width;
        left = clamp(left, EDGE_PADDING, window.innerWidth - panelRect.width - EDGE_PADDING);

        let top = buttonRect.bottom + GAP;
        if (top + panelRect.height > window.innerHeight - EDGE_PADDING) {
            top = buttonRect.top - panelRect.height - GAP;
        }
        top = clamp(top, EDGE_PADDING, window.innerHeight - panelRect.height - EDGE_PADDING);

        panel.style.left = `${left}px`;
        panel.style.top = `${top}px`;
        panel.style.visibility = "";
    }

    function keepPanelInViewport(panel) {
        const rect = panel.getBoundingClientRect();

        const left = clamp(
            rect.left,
            EDGE_PADDING,
            Math.max(EDGE_PADDING, window.innerWidth - rect.width - EDGE_PADDING)
        );

        const top = clamp(
            rect.top,
            EDGE_PADDING,
            Math.max(EDGE_PADDING, window.innerHeight - rect.height - EDGE_PADDING)
        );

        panel.style.left = `${left}px`;
        panel.style.top = `${top}px`;
    }

    function getAnimationGeometry(panel, button) {
        const panelRect = panel.getBoundingClientRect();
        const buttonRect = button.getBoundingClientRect();

        const fromX = buttonRect.right;
        const fromY = buttonRect.top + (buttonRect.height * 0.5);

        const toX = panelRect.right - 20;
        const toY = panelRect.top + 18;

        const translateX = fromX - toX;
        const translateY = fromY - toY;

        const scaleX = Math.max(buttonRect.width / panelRect.width, 0.06);
        const scaleY = Math.max(buttonRect.height / panelRect.height, 0.05);

        return { translateX, translateY, scaleX, scaleY, buttonRect };
    }

    function animateOpen(panel, button) {
        cancelActiveAnimation();

        const { translateX, translateY, scaleX, scaleY, buttonRect } =
            getAnimationGeometry(panel, button);

        activeAnimation = panel.animate(
            [
                {
                    opacity: 0.08,
                    transform: `translate(${translateX}px, ${translateY}px) scale(${scaleX}, ${scaleY})`,
                    borderRadius: `${Math.max(12, buttonRect.height / 2)}px`,
                    filter: "saturate(1.18)"
                },
                {
                    opacity: 1,
                    transform: "translate(0px, 0px) scale(1.055, 1.03)",
                    borderRadius: "28px",
                    filter: "saturate(1.04)",
                    offset: 0.68
                },
                {
                    opacity: 1,
                    transform: "translate(0px, 0px) scale(0.992, 0.998)",
                    borderRadius: "16px",
                    offset: 0.86
                },
                {
                    opacity: 1,
                    transform: "translate(0px, 0px) scale(1, 1)",
                    borderRadius: "18px",
                    filter: "saturate(1)"
                }
            ],
            {
                duration: OPEN_DURATION,
                easing: "cubic-bezier(0.16, 0.88, 0.22, 1.14)",
                fill: "forwards"
            }
        );

        activeAnimation.onfinish = function () {
            panel.style.opacity = "1";
            panel.style.transform = "translate(0px, 0px) scale(1, 1)";
            panel.style.borderRadius = "18px";
            panel.style.filter = "saturate(1)";
            activeAnimation = null;
        };

        activeAnimation.oncancel = function () {
            activeAnimation = null;
        };
    }

    function animateClose(panel, button, onFinish) {
        cancelActiveAnimation();

        const { translateX, translateY, scaleX, scaleY, buttonRect } =
            getAnimationGeometry(panel, button);

        activeAnimation = panel.animate(
            [
                {
                    opacity: 1,
                    transform: "translate(0px, 0px) scale(1, 1)",
                    borderRadius: "18px"
                },
                {
                    opacity: 0,
                    transform: `translate(${translateX}px, ${translateY}px) scale(${scaleX}, ${scaleY})`,
                    borderRadius: `${Math.max(12, buttonRect.height / 2)}px`
                }
            ],
            {
                duration: CLOSE_DURATION,
                easing: "cubic-bezier(0.55, 0.06, 0.68, 0.19)",
                fill: "forwards"
            }
        );

        activeAnimation.onfinish = function () {
            activeAnimation = null;
            onFinish();
        };

        activeAnimation.oncancel = function () {
            activeAnimation = null;
        };
    }

    function openPanel() {
        const button = getButton();
        if (!button) return;

        const panel = createPanel();
        attachControls(panel);

        if (panel.dataset.open === "true") {
            panel.style.zIndex = "2147483000";
            keepPanelInViewport(panel);
            return;
        }

        panel.dataset.open = "true";
        panel.setAttribute("aria-hidden", "false");

        showPanel(panel);

        if (!userMovedPanel) {
            placePanelFromButton(panel, button);
        } else {
            keepPanelInViewport(panel);
        }

        animateOpen(panel, button);
    }

    function closePanel() {
        const panel = getPanel();
        const button = getButton();

        if (!panel || panel.dataset.open !== "true") return;

        panel.dataset.open = "false";
        panel.setAttribute("aria-hidden", "true");

        if (!button) {
            cancelActiveAnimation();
            hidePanel(panel);
            return;
        }

        animateClose(panel, button, function () {
            hidePanel(panel);
        });
    }

    function togglePanel() {
        const panel = getPanel();
        if (panel && panel.dataset.open === "true") {
            closePanel();
        } else {
            openPanel();
        }
    }

    function enableDragging(panel) {
        const header = panel.querySelector(`#${HEADER_ID}`);

        header.addEventListener("pointerdown", function (event) {
            if (event.button !== 0) return;
            if (event.target.closest("button, input, select, textarea, label")) return;
            if (panel.dataset.open !== "true") return;

            const rect = panel.getBoundingClientRect();

            dragState = {
                pointerId: event.pointerId,
                startX: event.clientX,
                startY: event.clientY,
                left: rect.left,
                top: rect.top
            };

            userMovedPanel = true;
            header.classList.add("dragging");
            cancelActiveAnimation();

            if (typeof header.setPointerCapture === "function") {
                header.setPointerCapture(event.pointerId);
            }

            event.preventDefault();
        });

        header.addEventListener("pointermove", function (event) {
            if (!dragState || event.pointerId !== dragState.pointerId) return;

            const rect = panel.getBoundingClientRect();

            const nextLeft = clamp(
                dragState.left + (event.clientX - dragState.startX),
                EDGE_PADDING,
                window.innerWidth - rect.width - EDGE_PADDING
            );

            const nextTop = clamp(
                dragState.top + (event.clientY - dragState.startY),
                EDGE_PADDING,
                window.innerHeight - rect.height - EDGE_PADDING
            );

            panel.style.left = `${nextLeft}px`;
            panel.style.top = `${nextTop}px`;
        });

        function stopDragging(event) {
            if (!dragState) return;
            if (event.pointerId !== dragState.pointerId) return;

            dragState = null;
            header.classList.remove("dragging");

            if (
                typeof header.hasPointerCapture === "function" &&
                header.hasPointerCapture(event.pointerId)
            ) {
                header.releasePointerCapture(event.pointerId);
            }
        }

        header.addEventListener("pointerup", stopDragging);
        header.addEventListener("pointercancel", stopDragging);
    }

    function bindEvents() {
        document.addEventListener("click", function (event) {
            const button = event.target.closest(`#${BUTTON_ID}`);
            if (!button) return;

            event.preventDefault();
            togglePanel();
        });

        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape") {
                closePanel();
            }
        });

        window.addEventListener("resize", function () {
            const panel = getPanel();
            const button = getButton();

            if (!panel || panel.dataset.open !== "true") return;

            attachControls(panel);

            if (!userMovedPanel && button) {
                placePanelFromButton(panel, button);
                return;
            }

            keepPanelInViewport(panel);
        });

        document.addEventListener("shiny:value", function () {
            const panel = getPanel();
            if (!panel) return;
            attachControls(panel);
        });

        document.addEventListener("shiny:connected", function () {
            ensureControlsVisible();
            const panel = getPanel();
            if (panel) attachControls(panel);
        });
    }

    function init() {
        if (initialized) return;
        initialized = true;

        injectStyles();
        ensureControlsVisible();

        const panel = createPanel();
        attachControls(panel);
        startControlsObserver();
        bindEvents();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init, { once: true });
    } else {
        init();
    }
})();