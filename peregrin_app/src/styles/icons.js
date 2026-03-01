$(document).ready(function() {

  // Fallback copy method for non-HTTPS or older browsers
  function fallbackCopy(text) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.cssText = 'position:fixed;top:0;left:0;opacity:0;';
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    try {
      document.execCommand('copy');
      return true;
    } catch (err) {
      console.error('Fallback copy failed:', err);
      return false;
    } finally {
      document.body.removeChild(textarea);
    }
  }

  function copyToClipboard(text, iconElement) {
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(function() {
        iconElement.innerHTML = '&#x2713;';
        setTimeout(function() { iconElement.innerHTML = '&#x1F4CE;'; }, 1500);
      }).catch(function() {
        if (fallbackCopy(text)) {
          iconElement.innerHTML = '&#x2713;';
          setTimeout(function() { iconElement.innerHTML = '&#x1F4CE;'; }, 1500);
        }
      });
    } else {
      if (fallbackCopy(text)) {
        iconElement.innerHTML = '&#x2713;';
        setTimeout(function() { iconElement.innerHTML = '&#x1F4CE;'; }, 1500);
      }
    }
  }

  // Use MutationObserver to watch for dynamically added notifications
  const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      mutation.addedNodes.forEach(function(node) {
        if (node.nodeType !== 1) return;

        const notifications = node.classList && node.classList.contains('shiny-notification')
          ? [node]
          : Array.from(node.querySelectorAll('.shiny-notification') || []);

        notifications.forEach(function(notification) {
          if (notification.querySelector('.copy-icon')) return;

          const contentArea = notification.querySelector('.shiny-notification-content-text')
                           || notification.querySelector('.shiny-notification-content')
                           || notification;

          notification.style.position = 'relative';

          const copyIcon = document.createElement('span');
          copyIcon.innerHTML = '&#x1F4CE;';
          copyIcon.classList.add('copy-icon');
          copyIcon.style.cssText =
            'position:absolute; bottom:4px; right:4px; cursor:pointer; font-size:2em; z-index:99999; opacity:0.6; user-select:none;'
            + 'display:inline-flex; align-items:center; justify-content:center; width:1.2em; height:1.2em; text-align:center;';

          copyIcon.addEventListener('mouseenter', function() { this.style.opacity = '1'; });
          copyIcon.addEventListener('mouseleave', function() { this.style.opacity = '0.6'; });

          copyIcon.addEventListener('click', function(e) {
            e.stopPropagation();
            e.preventDefault();

            var cloned = contentArea.cloneNode(true);
            cloned.querySelectorAll('.copy-icon').forEach(function(el) { el.remove(); });
            cloned.querySelectorAll('.shiny-notification-close').forEach(function(el) { el.remove(); });

            var textToCopy = cloned.innerText || cloned.textContent || '';
            textToCopy = textToCopy.trim();

            if (textToCopy) {
              copyToClipboard(textToCopy, copyIcon);
            }
          });

          notification.appendChild(copyIcon);
        });
      });
    });
  });

  observer.observe(document.body, { childList: true, subtree: true });
});