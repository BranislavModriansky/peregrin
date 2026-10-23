(function () {
  const STATE = window.GRAPH_STATE || {};

  let lensCount = 0;
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const FISHEYE_STRENGTH = 0.05;      // glass magnification effect strength
  const BLUR_BACKDROP = 'blur(2px)';  // blur effect

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
    // so we resolve to the drawable leaves inside those groups. Elements under
    // a <defs> are excluded: matplotlib nests a shared marker/path definition
    // (e.g. one <path> reused by every scatter point's <use>) right inside the
    // gid group, and `querySelectorAll(DRAWABLE)` would otherwise pick it up
    // as an extra, non-rendered "node" — shifting every real node's index by
    // one wherever colors/edits are applied positionally (e.g. per-point
    // colormap sampling).
    function collect(id) {
      const roots = svg.querySelectorAll(`[id^="${id}"]`);
      const out = new Set();
      roots.forEach(node => {
        if (node.matches(DRAWABLE) && !node.closest('defs')) out.add(node);
        node.querySelectorAll(DRAWABLE).forEach(el => {
          if (!el.closest('defs')) out.add(el);
        });
      });
      return [...out];
    }

    // Hit targets = the original gid-tagged nodes (groups), so right-click works
    // even on the transparent group wrapper.
    const hitNodes = id => [...svg.querySelectorAll(`[id^="${id}"]`)];

    const tip = document.createElement('div');
    tip.className = 'pg-tooltip';
    root.appendChild(tip);

    const lens = createFisheyeLens(tip);

    // Last panel position/focus, so Restore can re-render it in place.
    let panelState = null;

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
      panelState = { x: parseFloat(tip.style.left) || 0, y: parseFloat(tip.style.top) || 0 };

      if (!drag || e.pointerId !== drag.id) return;
      tip.releasePointerCapture(e.pointerId);
      tip.classList.remove('pg-dragged');
      drag = null;
    };
    tip.addEventListener('pointerup', endDrag);
    tip.addEventListener('pointercancel', endDrag);

    // ---- attribute editing helpers (mutate SVG in place) ----
    // Only override an attribute if it isn't explicitly "none" (preserves
    // fill-only / stroke-only artists like unfilled head scatters).
    function setStrokeColor(nodes, v) {
      nodes.forEach(p => {
        if (p.getAttribute('stroke') !== 'none') p.setAttribute('stroke', v);
        if (p.style && p.style.stroke && p.style.stroke !== 'none') p.style.stroke = v;
      });
    }
    function setFillColor(nodes, v) {
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

      // Resolve the scatter paths referenced by Matplotlib's <use> elements.
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
          // Matplotlib scatter definitions normally use (0, 0) as their center.
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

      // Fallback for SVG scatters rendered directly instead of through <use>.
      nodes
        .filter(node => !node.closest('defs'))
        .forEach(scaleAroundCenter);
    }

    // ---- control archetypes -------------------------------------------------
    // A gid in the payload declares a `type` (line | scatter | grid |
    // background) and an `id` (matching its `pg:<id>` gid). The archetype maps
    // that type to a `paint` function (how a color is applied — stroke, fill or
    // both) plus the non-color `controls` the panel renders. Color itself is
    // rendered generically from the gid's `color` descriptor (single /
    // categorical / numeric), so any graph becomes editable just by emitting
    // gids — nothing here is specific to tracks.
    const hex = (c, fallback) => (c || fallback).slice(0, 7); // color input needs #rrggbb

    const rangeControl = (key, label, func, min = 0, max = 2, step = 0.25) => ({ 
      key, label, 
      input: { 
        type: 'range', 
        min: min, 
        max: max, 
        step: step
      }, 
      apply: func
    });

    const ARCHETYPES = {
      line: {
        paint: setStrokeColor,
        controls: label => [rangeControl('width', `${label} width`, setStrokeWidth)],
      },
      scatter: {
        paint: (n, v) => {
          setStrokeColor(n, v);
          setFillColor(n, v);
        },
        controls: label => [rangeControl('size', `${label} size`, setScale, 0, 3, 0.25)],
      },
      grid: {
        paint: setStrokeColor,
        controls: label => [rangeControl('width', `${label} width`, setStrokeWidth)],
      },
      background: {
        paint: setFillColor,
        controls: () => [],
      },
    };

    // ---- shared palette registry (for numeric / colormap gids) ----
    const CMAPS = (STATE.cmaps) || { quantitative: [], samples: {} };

    function hexToRgb(h) {
      h = (h || '').replace('#', '');
      return [parseInt(h.slice(0, 2), 16) || 0, parseInt(h.slice(2, 4), 16) || 0, parseInt(h.slice(4, 6), 16) || 0];
    }
    function rgbToHex(r, g, b) {
      const c = v => ('0' + Math.round(v).toString(16)).slice(-2);
      return '#' + c(r) + c(g) + c(b);
    }
    // Interpolate a colormap (list of hex samples) at t in [0, 1].
    function sampleCmap(samples, t) {
      if (!samples || !samples.length) return '#000000';
      if (samples.length === 1) return samples[0];
      const x = Math.min(1, Math.max(0, t)) * (samples.length - 1);
      const i = Math.floor(x);
      if (i >= samples.length - 1) return samples[samples.length - 1];
      const a = hexToRgb(samples[i]);
      const b = hexToRgb(samples[i + 1]);
      const f = x - i;
      return rgbToHex(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f);
    }

    const clone = obj => (obj ? JSON.parse(JSON.stringify(obj)) : obj);

    // Resolve declared gids to live SVG nodes + their controls/color. gids
    // whose type is unknown, or whose elements aren't present, are skipped.
    // STATE.style.gid is a dict keyed by gid suffix (e.g. 'tracks', 'heads').
    const gids = Object.entries((STATE.style && STATE.style.gid) || {})
      .map(([suffix, gid]) => {
        const arch = ARCHETYPES[gid.type] || { paint: setStrokeColor, controls: () => [] };
        return {
          suffix,
          type: gid.type,
          label: gid.label,
          paint: arch.paint,
          controls: arch.controls(gid.label),
          color: clone(gid.color) || null,
          values: Object.assign({}, gid.values),
          nodes: collect('pg:' + suffix),
          hits: hitNodes('pg:' + suffix),
        };
      })
      .filter(gid => (gid.controls.length || gid.color) && (gid.nodes.length || gid.hits.length));

    const gidBySuffix = new Map(gids.map(gid => [gid.suffix, gid]));

    // Immutable snapshot of each gid's initial state for Refresh/restore.
    const ORIGINAL = new Map(
      gids.map(gid => [gid.suffix, { values: Object.assign({}, gid.values), color: clone(gid.color) }]));

    // gids whose color is `{ colortype: 'matching', target: <suffix> }` mirror
    // another gid's color instead of exposing their own color row. Indexed by
    // target suffix so editing a gid's color can re-paint every follower.
    const MATCHING_TO_TARGET = new Map();
    gids.forEach(gid => {
      if (gid.color && gid.color.colortype === 'matching') {
        const followers = MATCHING_TO_TARGET.get(gid.color.target) || [];
        followers.push(gid);
        MATCHING_TO_TARGET.set(gid.color.target, followers);
      }
    });

    // ---- color application (mode-aware) ----
    const categoryNodes = (gid, token) => collect('pg:' + gid.suffix + ':' + token);

    // Paint a gid whose color is matching to another gid's color descriptor.
    // Quantitative targets are sampled per-node using the follower's own
    // `values` (e.g. a track head's own end-of-track position), so each
    // follower node gets the color matching its own position on the target's
    // colormap rather than one shared color. Uniform / qualitative targets
    // have no such per-node granularity, so the follower is simply painted
    // with the target's single / first-matching color.
    function applyMatchingColor(gid) {
      const target = gidBySuffix.get(gid.color.target);
      const tc = target && target.color;
      if (!tc) return;
      if (tc.colortype === 'uniform') {
        gid.paint(gid.nodes, tc.value);
      } else if (tc.colortype === 'qualitative') {
        const own = (gid.color.categories || []).length ? gid.color.categories : tc.categories;
        own.forEach(cat => {
          const match = tc.categories.find(c => c.token === cat.token) || tc.categories[0];
          if (match) gid.paint(categoryNodes(gid, cat.token), match.color);
        });
      } else if (tc.colortype === 'quantitative') {
        const samples = CMAPS.samples[tc.cmap];
        const vals = gid.color.values || [];
        gid.nodes.forEach((node, i) => gid.paint([node], sampleCmap(samples, vals[i] ?? 0)));
      }
    }

    function applygidColor(gid) {
      const c = gid.color;
      if (!c) return;
      if (c.colortype === 'uniform') {
        gid.paint(gid.nodes, c.value);
      } else if (c.colortype === 'qualitative') {
        c.categories.forEach(cat => gid.paint(categoryNodes(gid, cat.token), cat.color));
      } else if (c.colortype === 'quantitative') {
        const samples = CMAPS.samples[c.cmap];
        const vals = c.values || [];
        gid.nodes.forEach((node, i) => gid.paint([node], sampleCmap(samples, vals[i] ?? 0)));
      } else if (c.colortype === 'matching') {
        applyMatchingColor(gid);
      }
    }

    // Re-paint every gid matching to `gidSuffix`'s color (called whenever that
    // gid's own color changes), so e.g. track heads stay matched to the
    // track's current color/colormap without their own editable color row.
    function propagateMatchingColors(gidSuffix) {
      (MATCHING_TO_TARGET.get(gidSuffix) || []).forEach(applygidColor);
    }

    function applyColor(gidSuffix, colortype, token, value) {
      const gid = gidBySuffix.get(gidSuffix);
      if (!gid || !gid.color) return;
      if (colortype === 'uniform') {
        gid.color.value = value;
        gid.paint(gid.nodes, value);
      } else if (colortype === 'qualitative') {
        const cat = gid.color.categories.find(c => c.token === token);
        if (cat) { cat.color = value; gid.paint(categoryNodes(gid, token), value); }
      } else if (colortype === 'quantitative') {
        gid.color.cmap = value;
        applygidColor(gid);
      }
      propagateMatchingColors(gidSuffix);
    }

    // ---- colormap combobox (filterable typeahead) ----
    // A text input backed by an always-open options panel. Typing filters the
    // colormaps by prefix; the top match is kept "semi-selected" so Enter commits
    // it. Anything that isn't a real colormap name is ignored (revert to the last
    // valid value).
    function setupCombo(combo) {
      const gidSuffix = combo.dataset.gid;
      const input = combo.querySelector('.pg-combo-input');
      const list = combo.querySelector('.pg-combo-options');
      const options = [...list.querySelectorAll('.pg-combo-option')];
      const names = options.map(o => o.dataset.value);
      let active = -1;

      const currentCmap = () => (gidBySuffix.get(gidSuffix).color || {}).cmap || '';
      const visible = () => options.filter(o => o.style.display !== 'none');

      function setActive(i) {
        const vis = visible();
        options.forEach(o => o.classList.remove('pg-combo-option--active'));
        if (!vis.length) { active = -1; return; }
        active = Math.max(0, Math.min(i, vis.length - 1));
        vis[active].classList.add('pg-combo-option--active');
        vis[active].scrollIntoView({ block: 'nearest' });
      }

      // Prefix filter; first match floats to the top and is semi-selected.
      function filter() {
        const q = input.value.trim().toLowerCase();
        options.forEach(o => {
          o.style.display = o.dataset.value.toLowerCase().startsWith(q) ? '' : 'none';
        });
        list.scrollTop = 0;
        setActive(0);
      }

      function open() { combo.classList.add('pg-combo--open'); }

      function revert() {
        combo.classList.remove('pg-combo--open');
        input.value = currentCmap();
      }

      function commit(value) {
        if (value && names.includes(value)) {
          applyColor(gidSuffix, 'quantitative', null, value);
          input.value = value;
          combo.classList.remove('pg-combo--open');
          return true;
        }
        return false;
      }

      // Clear so the user sees every colormap and can type fresh; the current
      // colormap stays visible as a placeholder for reference. Bound to both
      // 'focus' and 'click': after a mouse selection the input keeps focus
      // (mousedown on an option prevents the default blur), so a follow-up
      // click on the already-focused input would otherwise fire no 'focus'
      // event and never clear/reopen the list.
      function activate() {
        open();
        input.placeholder = currentCmap();
        input.value = '';
        filter();
      }
      input.addEventListener('focus', activate);
      input.addEventListener('click', activate);
      input.addEventListener('input', () => { open(); filter(); });
      input.addEventListener('keydown', e => {
        if (e.key === 'ArrowDown') { e.preventDefault(); open(); setActive(active + 1); }
        else if (e.key === 'ArrowUp') { e.preventDefault(); open(); setActive(active - 1); }
        else if (e.key === 'Enter') {
          e.preventDefault();
          const vis = visible();
          if (active >= 0 && vis[active]) { commit(vis[active].dataset.value); input.blur(); }
          else revert();
        } else if (e.key === 'Escape') { e.preventDefault(); revert(); input.blur(); }
      });
      // Delay so a mousedown on an option commits before blur reverts.
      input.addEventListener('blur', () => setTimeout(() => {
        if (!combo.classList.contains('pg-combo--open')) return;
        if (!commit(input.value.trim())) revert();
      }, 150));

      options.forEach(o => {
        o.addEventListener('mousedown', e => { e.preventDefault(); commit(o.dataset.value); });
        o.addEventListener('mouseenter', () => setActive(visible().indexOf(o)));
      });
    }

    // Non-color controls (width, size): applied uniformly to the gid's nodes.
    function applyEdit(gidSuffix, key, value) {
      const gid = gidBySuffix.get(gidSuffix);
      if (!gid) return;
      const control = gid.controls.find(c => c.key === key);
      if (!control) return;
      control.apply(gid.nodes, value);
      gid.values[key] = value;
    }

    // ---- export / restore helpers ----
    function triggerDownload(blob, filename) {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      // Stop the synthetic click from bubbling to document, where the
      // "click outside closes the tooltip" listener would otherwise fire.
      a.addEventListener('click', ev => ev.stopPropagation());
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    // Clone the live SVG (reflecting any edits) and make it standalone.
    function serializeSvg() {
      const clone = svg.cloneNode(true);
      clone.setAttribute('xmlns', SVG_NS);
      clone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
      return new XMLSerializer().serializeToString(clone);
    }

    function svgPixelSize() {
      const rect = svg.getBoundingClientRect();
      let w = rect.width;
      let h = rect.height;
      if (!w || !h) {
        const vb = svg.getAttribute('viewBox');
        if (vb) {
          const parts = vb.split(/[ ,]+/).map(Number);
          w = parts[2];
          h = parts[3];
        }
      }
      return { w: Math.max(1, Math.round(w)), h: Math.max(1, Math.round(h)) };
    }

    function exportSVG() {
      const blob = new Blob([serializeSvg()], {
        type: 'image/svg+xml;charset=utf-8',
      });
      triggerDownload(blob, 'graph.svg');
    }

    function exportPNG() {
      const { w, h } = svgPixelSize();
      const scale = 2; // render at 2x for a crisper raster
      const source = serializeSvg();
      const svgUrl =
        'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(source);
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = w * scale;
        canvas.height = h * scale;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(blob => {
          if (blob) triggerDownload(blob, 'graph.png');
        }, 'image/png');
      };
      img.src = svgUrl;
    }

    // Restore every editable attribute to the value it had on first render.
    // Done in two passes: reset every gid's color descriptor first, then
    // paint. A matching gid (e.g. track heads) reads its target's descriptor
    // while painting, so the target must already hold its restored color
    // regardless of which gid happens to come first in `gids`.
    function restore() {
      gids.forEach(gid => {
        const original = ORIGINAL.get(gid.suffix) || {};
        const values = original.values || {};
        gid.controls.forEach(control => {
          if (!(control.key in values)) return;
          const value = values[control.key];
          control.apply(gid.nodes, value);
          gid.values[control.key] = value;
        });
        if (original.color) gid.color = clone(original.color);
      });
      gids.forEach(gid => { if (gid.color) applygidColor(gid); });

      // Re-render the open panel so every input reflects the restored state.
      openPanel(panelState.x, panelState.y);
    }

    // ---- tooltip UI ----
    const header       = `<div class="pg-header">⋮⋮ <i>tooltip</i></div>`;
    const closeButton  = `<span class="pg-close">✕</span>`;
    const exportIcon =
      `<svg class="pg-btn-icon" xmlns="http://www.w3.org/2000/svg" height="1rem" viewBox="0 -960 960 960" width="1rem" fill="currentColor"><path d="M480-480ZM202-65l-56-57 118-118h-90v-80h226v226h-80v-89L202-65Zm278-15v-80h240v-440H520v-200H240v400h-80v-400q0-33 23.5-56.5T240-880h320l240 240v480q0 33-23.5 56.5T720-80H480Z"/></svg>`;
    const refreshIcon =
      `<svg class="pg-btn-icon" xmlns="http://www.w3.org/2000/svg" height="1rem" viewBox="0 -960 960 960" width="1rem" fill="currentColor"><path d="M480-160q-134 0-227-93t-93-227q0-134 93-227t227-93q69 0 132 28.5T720-690v-110h80v280H520v-80h168q-32-56-87.5-88T480-720q-100 0-170 70t-70 170q0 100 70 170t170 70q77 0 139-44t87-116h84q-28 106-114 173t-196 67Z"/></svg>`;
    const graphSubmenu = 
      `<div class="pg-graph-submenu">
        <button type="button" data-act="svg">${exportIcon}SVG</button> <button type="button" data-act="png">${exportIcon}PNG</button> <button type="button" data-act="restore" title="Restore">${refreshIcon}</button>
      </div>`;
    const row = (label, input) =>
      `<div class="pg-row"><label>${label}</label><br>${input}</div>`;


    function showTip(x, y, html) {
      tip.innerHTML = html;
      tip.style.left = x + 'px';
      tip.style.top = y + 'px';
      tip.style.display = 'block';
      lens.update();
      tip.querySelectorAll('[data-gid][data-k]').forEach(el => {
        el.addEventListener('input', () => {
          let v = el.value;
          if (el.type === 'range') v = parseFloat(v);
          applyEdit(el.dataset.gid, el.dataset.k, v);
        });
      });
      // Single / per-category color pickers (plain inputs with data-cmode).
      tip.querySelectorAll('input[data-gid][data-colortype]').forEach(el => {
        const handler = () => applyColor(el.dataset.gid, el.dataset.colortype, el.dataset.token, el.value);
        el.addEventListener('input', handler);
        el.addEventListener('change', handler);
      });
      // Colormap comboboxes (filterable typeahead).
      tip.querySelectorAll('.pg-combo[data-colortype="quantitative"]').forEach(setupCombo);
      const close = tip.querySelector('.pg-close');
      if (close) close.onclick = () => (tip.style.display = 'none');
      tip.querySelectorAll('[data-act]').forEach(btn => {
        btn.onclick = e => {
          // Stop this click from reaching the document "click outside
          // closes the tooltip" listener. Restore re-renders the panel
          // mid-bubble, which would otherwise make the listener see a
          // detached target and close the freshly rendered tooltip.
          e.stopPropagation();
          const act = btn.dataset.act;
          if (act === 'svg') exportSVG();
          if (act === 'png') exportPNG();
          if (act === 'restore') restore();
        };
      });
    }

    // ---- panel assembly (data-driven) ----
    // Non-color controls (range sliders) from the archetype descriptor.
    function inputHtml(gid, control) {
      const value = gid.values[control.key];
      const base = `data-gid="${gid.suffix}" data-k="${control.key}"`;
      if (control.input.type === 'color') {
        return `<input type="color" ${base} value="${hex(value, '#000000')}">`;
      }
      const { min = 0, max = 1, step = 0.1 } = control.input;
      return `<input type="range" min="${min}" max="${max}" step="${step}" ${base} value="${value ?? min}">`;
    }

    // Color control, rendered from the gid's color descriptor mode.
    function colorHtml(gid) {
      const c = gid.color;
      if (!c) return '';
      if (c.colortype === 'uniform') {
        return row(gid.label + ' color',
          `<input type="color" data-gid="${gid.suffix}" data-colortype="uniform" value="${hex(c.value, '#000000')}">`);
      }
      if (c.colortype === 'qualitative') {
        return (c.categories || [])
          .map(cat => row(cat.label,
            `<input type="color" data-gid="${gid.suffix}" data-colortype="qualitative" data-token="${cat.token}" value="${hex(cat.color, '#000000')}">`))
          .join('');
      }
      if (c.colortype === 'quantitative') {
        const options = (CMAPS.quantitative || [])
          .map(name => `<div class="pg-combo-option" data-value="${name}">${name}</div>`)
          .join('');
        return row(gid.label + ' color-map',
          `<div class="pg-combo" data-gid="${gid.suffix}" data-colortype="quantitative">
            <input class="pg-combo-input" type="text" spellcheck="false" autocomplete="off" value="${c.cmap}">
            <div class="pg-combo-options">${options}</div>
          </div>`
        );
      }
      return '';
    }

    // One unified panel with a titled section per gid. The right-clicked
    // gid floats to the top so its controls line up with what was clicked.
    function openPanel(x, y) {
      panelState = { x, y };
      const body = gids.map(
        gid => colorHtml(gid) + gid.controls.map(
          control => row(control.label, inputHtml(gid, control))
        ).join('')
      ).join('');
      showTip(x, y, closeButton + header + graphSubmenu + body);
    }

    // ---- event wiring: right-click on a tagged element opens the panel ----
    function bind(nodes) {
      nodes.forEach(n => {
        n.style.pointerEvents = 'all';
        n.addEventListener('contextmenu', e => {
          e.preventDefault();
          e.stopPropagation();
          const r = root.getBoundingClientRect();
          openPanel(e.clientX - r.left, e.clientY - r.top);
        });
      });
    }

    gids.forEach(gid =>
      bind(gid.hits.length ? gid.hits : gid.nodes));

    svg.addEventListener('contextmenu', e => e.preventDefault());

    // Clicking anywhere outside the panel closes it. Uses 'click' (left button),
    // so the right-click that opens the panel never triggers a close.
    document.addEventListener('click', e => {
      if (tip.style.display === 'block' && !tip.contains(e.target)) {
        tip.style.display = 'none';
      }
    });
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

      const value = `url(#${id}) ${BLUR_BACKDROP}`;
      tip.style.backdropFilter = value;
      tip.style.webkitBackdropFilter = value;
    }

    if (window.ResizeObserver) new ResizeObserver(update).observe(tip);
    return { update };
  }
})();