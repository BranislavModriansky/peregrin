(function () {
  const S = window.GRAPH_STATE || {};

  // Declared before init() runs; const/let are not hoisted.
  let lensCount = 0;
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const FISHEYE_STRENGTH = 0.05; // 0 = flat glass, <1 = centre magnification 1/(1-s)
  const BACKDROP_BLUR = 'blur(2px)';

  // ---- inject CSS for the tooltip --------------------
  if (!document.getElementById('peregrin-graph-style')) {
    const style = document.createElement('style');
    style.id = 'peregrin-graph-style';
    style.textContent = "__PEREGRIN_TOOLTIP_CSS__";
    document.head.appendChild(style);
  }

  document.querySelectorAll('.peregrin-graph:not([data-pg-init])').forEach(init);

  function init(root) {
    root.dataset.pgInit = '1';
    const svg = root.querySelector('svg');
    if (!svg) return;

    const DRAWABLE = 'path,line,rect,circle,polygon,polyline,use';

    // Collect every element whose own id starts with `prefix`, OR that lives
    // inside a <g> whose id starts with `prefix`. Matplotlib emits gids as ids
    // on wrapping <g> groups (and sometimes appends suffixes / dedup numbers),
    // so we resolve to the drawable leaves inside those groups.
    function collect(prefix) {
      const roots = svg.querySelectorAll(`[id^="${prefix}"]`);
      const out = new Set();
      roots.forEach(node => {
        if (node.matches(DRAWABLE)) out.add(node);
        node.querySelectorAll(DRAWABLE).forEach(el => out.add(el));
      });
      return [...out];
    }

    const groups = {
      tracks: collect('pg:tracks'),
      heads: collect('pg:heads'),
      grid: collect('pg:grid'),
      background: collect('pg:background'),
      figure: collect('pg:figure')
    };

    // Hit targets = the original gid-tagged nodes (groups), so right-click works
    // even on the transparent group wrapper.
    const hitNodes = prefix => [...svg.querySelectorAll(`[id^="${prefix}"]`)];

    const tip = document.createElement('div');
    tip.className = 'pg-tooltip';
    root.appendChild(tip);

    const lens = createFisheyeLens(tip);

    // ---- drag the panel by its header ----
    let drag = null;
    tip.addEventListener('pointerdown', e => {
      if (e.button !== 0 || !e.target.closest('.pg-header')) return;
      drag = {
        id: e.pointerId,
        x: e.clientX,
        y: e.clientY,
        left: parseFloat(tip.style.left) || 0,
        top: parseFloat(tip.style.top) || 0,
      };
      tip.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    tip.addEventListener('pointermove', e => {
      if (!drag || e.pointerId !== drag.id) return;
      tip.style.left = drag.left + e.clientX - drag.x + 'px';
      tip.style.top = drag.top + e.clientY - drag.y + 'px';
      tip.classList.add('pg-dragged');
    });
    const endDrag = e => {
      if (!drag || e.pointerId !== drag.id) return;
      tip.releasePointerCapture(e.pointerId);
      tip.classList.remove('pg-dragged');
      drag = null;
    };
    tip.addEventListener('pointerup', endDrag);
    tip.addEventListener('pointercancel', endDrag);

    // ---- attribute editing helpers (mutate SVG in place) ----
    // Only override an attribute if it isn't explicitly "none" (preserves
    // fill-only / stroke-only artists like unfilled head markers).
    function setStroke(nodes, v) {
      nodes.forEach(p => {
        if (p.getAttribute('stroke') !== 'none') p.setAttribute('stroke', v);
        // matplotlib sometimes puts color in inline style
        if (p.style && p.style.stroke && p.style.stroke !== 'none') p.style.stroke = v;
      });
    }
    function setFill(nodes, v) {
      nodes.forEach(p => {
        if (p.getAttribute('fill') !== 'none') p.setAttribute('fill', v);
        if (p.style && p.style.fill && p.style.fill !== 'none') p.style.fill = v;
      });
    }
    function setStrokeWidth(nodes, v) {
      nodes.forEach(p => {
        p.setAttribute('stroke-width', v);
        if (p.style && p.style.strokeWidth) p.style.strokeWidth = v;
      });
    }
    function setScale(nodes, value) {
      const scale = Number(value);
      if (!Number.isFinite(scale) || scale <= 0) return;

      const uses = nodes.filter(
        node => node.localName === 'use' && !node.closest('defs')
      );

      // Resolve the marker paths referenced by Matplotlib's <use> elements.
      // Scaling the definition preserves every <use x="..." y="..."> position.
      const definitions = new Set();

      uses.forEach(use => {
        const href =
          use.getAttribute('href') ||
          use.getAttribute('xlink:href') ||
          use.getAttributeNS('http://www.w3.org/1999/xlink', 'href');

        if (!href || !href.startsWith('#')) return;

        const id = href.slice(1);
        const definition = [...svg.querySelectorAll('defs [id]')]
          .find(node => node.id === id);

        if (definition) definitions.add(definition);
      });

      function scaleAroundCenter(node) {
        if (node.dataset.pgBaseTransform === undefined) {
          node.dataset.pgBaseTransform =
            node.getAttribute('transform') || '';
        }

        let cx = 0;
        let cy = 0;

        try {
          const box = node.getBBox();
          cx = box.x + box.width / 2;
          cy = box.y + box.height / 2;
        } catch (_) {
          // Matplotlib marker definitions normally use (0, 0) as their center.
        }

        const scaling =
          `translate(${cx} ${cy}) scale(${scale}) translate(${-cx} ${-cy})`;

        const transform = [
          node.dataset.pgBaseTransform,
          scaling,
        ].filter(Boolean).join(' ');

        node.setAttribute('transform', transform);
      }

      if (definitions.size) {
        definitions.forEach(scaleAroundCenter);
        return;
      }

      // Fallback for SVG markers rendered directly instead of through <use>.
      nodes
        .filter(node => !node.closest('defs'))
        .forEach(scaleAroundCenter);
    }

    const EDITORS = {
      tracks: {
        color: v => setStroke(groups.tracks, v),
        lw: v => setStrokeWidth(groups.tracks, v),
        headColor: v => {
          setStroke(groups.heads, v);
          setFill(groups.heads, v);
        },
        headSize: v => setScale(groups.heads, v),
      },
      grid: {
        gridColor: v => setStroke(groups.grid, v),
        gridLw: v => setStrokeWidth(groups.grid, v),
      },
      background: {
        faceColor: v => {
          setFill(groups.background, v);
        },
      },
    };

    function applyEdit(grp, key, value) {
      const fn = EDITORS[grp] && EDITORS[grp][key];
      if (fn) fn(value);
      if (S.style) S.style[key] = value;
    }

    // ---- tooltip UI ----
    const header = t =>
      `<div class="pg-header">${t}</div>`;
    const closeButton = `<span class="pg-close">✕</span>`;
    const row = (label, input) =>
      `<div class="pg-row"><label>${label}</label><br>${input}</div>`;


    function showTip(x, y, html) {
      tip.innerHTML = html;
      tip.style.left = x + 'px';
      tip.style.top = y + 'px';
      tip.style.display = 'block';
      lens.update();
      tip.querySelectorAll('[data-k]').forEach(el => {
        el.addEventListener('input', () => {
          let v = el.value;
          if (el.type === 'range') v = parseFloat(v);
          applyEdit(el.dataset.grp, el.dataset.k, v);
        });
      });
      const close = tip.querySelector('.pg-close');
      if (close) close.onclick = () => (tip.style.display = 'none');
    }

    const st = S.style || {};
    const hex = (c, fallback) => (c || fallback).slice(0, 7); // color input needs #rrggbb

    const MENUS = {
      figure: (x, y) => showTip(x, y, closeButton + header('⋮⋮ <i>tooltip</i>')
        + row('Color', `<input type="color" data-grp="tracks" data-k="color" value="${hex(st.trackColor, '#000000')}">`)
        + row('Head color', `<input type="color" data-grp="tracks" data-k="headColor" value="${hex(st.trackColor, '#000000')}">`)
        + row('Line width', `<input type="range" data-grp="tracks" data-k="lw" min="0.25" max="6" step="0.25" value="${st.lw ?? 1}">`)
        + row('Head size', `<input type="range" data-grp="tracks" data-k="headSize" min="0.25" max="6" step="0.25" value="${st.headSize ?? 1}">`)
        + row('Grid color', `<input type="color" data-grp="grid" data-k="gridColor" value="${hex(st.gridColor, '#dcdcdc')}">`)
        + row('Grid width', `<input type="range" data-grp="grid" data-k="gridLw" min="0" max="4" step="0.25" value="${st.gridLw ?? 0.75}">`)
        + row('background color', `<input type="color" data-grp="background" data-k="faceColor" value="${hex(st.faceColor, '#ffffff')}">`)),
    };

    // ---- event wiring: right-click on the tagged element opens its menu ----
    function bind(nodes, menu) {
      nodes.forEach(n => {
        n.style.pointerEvents = 'all';
        n.addEventListener('contextmenu', e => {
          e.preventDefault();
          e.stopPropagation();
          const r = root.getBoundingClientRect();
          MENUS[menu](e.clientX - r.left, e.clientY - r.top);
        });
      });
    }

    bind(hitNodes('pg:tracks'), 'figure');
    bind(hitNodes('pg:heads'), 'figure');
    bind(hitNodes('pg:grid'), 'figure');
    // background last so figure/background wrappers both open the bg menu
    bind(hitNodes('pg:background'), 'figure');

    svg.addEventListener('contextmenu', e => e.preventDefault());
  }

  // ---- fisheye lens for the light transmitted through the panel ----
  // An SVG feDisplacementMap is applied as a backdrop-filter. The displacement
  // map is a radial barrel distortion rendered to a canvas sized to the panel,
  // so the background is magnified at the centre and compressed towards the rim.
  function createFisheyeLens(tip) {
    const noop = { update() {} };
    const supported =
      window.CSS && CSS.supports &&
      (CSS.supports('backdrop-filter', 'url(#pg) blur(1px)') ||
       CSS.supports('-webkit-backdrop-filter', 'url(#pg) blur(1px)'));
    if (!supported) return noop;

    const id = `pg-fisheye-${++lensCount}`;
    const defs = document.createElementNS(SVG_NS, 'svg');
    defs.setAttribute('aria-hidden', 'true');
    defs.setAttribute('width', '0');
    defs.setAttribute('height', '0');
    defs.style.cssText = 'position:absolute;width:0;height:0;overflow:hidden;pointer-events:none';
    defs.innerHTML =
      `<filter id="${id}" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse"` +
      ` color-interpolation-filters="sRGB" x="0" y="0" width="1" height="1">` +
      `<feImage result="map" x="0" y="0" width="1" height="1" preserveAspectRatio="none"/>` +
      `<feDisplacementMap in="SourceGraphic" in2="map" scale="0"` +
      ` xChannelSelector="R" yChannelSelector="G"/>` +
      `</filter>`;
    document.body.appendChild(defs);

    const filter = defs.querySelector('filter');
    const image = defs.querySelector('feImage');
    const displace = defs.querySelector('feDisplacementMap');
    let size = '';

    function update() {
      const w = Math.round(tip.offsetWidth);
      const h = Math.round(tip.offsetHeight);
      if (!w || !h || size === `${w}x${h}`) return;
      size = `${w}x${h}`;

      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      const data = ctx.createImageData(w, h);
      const hw = w / 2;
      const hh = h / 2;
      // Largest displacement of r*(1-r^2)*s occurs at r = 1/sqrt(3).
      const maxShift = FISHEYE_STRENGTH * (2 / (3 * Math.sqrt(3))) * Math.max(hw, hh);
      const scale = Math.max(1, Math.ceil(maxShift * 2));

      for (let y = 0; y < h; y++) {
        const ny = (y + 0.5 - hh) / hh;
        for (let x = 0; x < w; x++) {
          const nx = (x + 0.5 - hw) / hw;
          const r2 = nx * nx + ny * ny;
          // Sample closer to the centre: offset = -s * (1 - r^2) * n.
          const k = r2 < 1 ? -FISHEYE_STRENGTH * (1 - r2) : 0;
          const i = (y * w + x) * 4;
          data.data[i] = Math.round(255 * (0.5 + (k * nx * hw) / scale));
          data.data[i + 1] = Math.round(255 * (0.5 + (k * ny * hh) / scale));
          data.data[i + 2] = 128;
          data.data[i + 3] = 255;
        }
      }
      ctx.putImageData(data, 0, 0);

      [filter, image].forEach(el => {
        el.setAttribute('width', w);
        el.setAttribute('height', h);
      });
      const url = canvas.toDataURL();
      image.setAttribute('href', url);
      image.setAttributeNS('http://www.w3.org/1999/xlink', 'xlink:href', url);
      displace.setAttribute('scale', scale);

      const value = `url(#${id}) ${BACKDROP_BLUR}`;
      tip.style.backdropFilter = value;
      tip.style.webkitBackdropFilter = value;
    }

    if (window.ResizeObserver) new ResizeObserver(update).observe(tip);
    return { update };
  }
})();