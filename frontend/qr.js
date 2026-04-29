/**
 * StructScan — QR Code Panel
 * Shows WiFi QR code when Live Camera is selected.
 * Mobile devices scan it to connect over same WiFi — no internet needed.
 */

(function () {
  var QR_PORT = 8000;

  function removeQR() {
    var el = document.getElementById('structscan-qr');
    if (el) el.remove();
  }

  function showQR() {
    removeQR();

    var panel = document.createElement('div');
    panel.id  = 'structscan-qr';
    panel.style.cssText = [
      'position:fixed',
      'bottom:24px',
      'right:24px',
      'z-index:9999',
      'background:#ffffff',
      'border:1px solid #e8e5e0',
      'padding:16px 16px 12px',
      'box-shadow:0 4px 24px rgba(0,0,0,.10)',
      'text-align:center',
      'width:164px',
      'font-family:"DM Mono",monospace,sans-serif',
    ].join(';');

    panel.innerHTML = [
      '<div style="font-size:9px;letter-spacing:.15em;color:#8a8680;',
          'text-transform:uppercase;margin-bottom:10px">',
        '📱 OPEN ON MOBILE',
      '</div>',
      '<img src="/qr?port=' + QR_PORT + '"',
          ' id="structscan-qr-img"',
          ' width="132" height="132"',
          ' style="display:block;border:1px solid #f0eeea"',
          ' onerror="this.style.display=\'none\';',
              'document.getElementById(\'structscan-qr-err\').style.display=\'block\'"',
          ' alt="QR Code" />',
      '<div id="structscan-qr-err" style="display:none;font-size:10px;',
          'color:#dc2626;margin:8px 0">',
        'Run:<br><code>pip install qrcode pillow</code>',
      '</div>',
      '<div id="structscan-qr-url" style="font-size:9px;color:#3a3835;',
          'margin-top:8px;word-break:break-all;line-height:1.4">',
        'Loading URL...',
      '</div>',
      '<div style="font-size:8px;color:#b8b4ae;margin-top:4px">',
        'Same WiFi · No internet',
      '</div>',
      '<button onclick="document.getElementById(\'structscan-qr\').remove()"',
          ' style="margin-top:10px;font-size:9px;padding:3px 12px;',
              'border:1px solid #e8e5e0;background:none;cursor:pointer;',
              'color:#8a8680;letter-spacing:.05em">',
        '✕ CLOSE',
      '</button>',
    ].join('');

    document.body.appendChild(panel);

    // Fetch real URL to display
    fetch('/network-url?port=' + QR_PORT)
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var el = document.getElementById('structscan-qr-url');
        if (el) el.textContent = d.url;
      })
      .catch(function () {
        var el = document.getElementById('structscan-qr-url');
        if (el) el.textContent = 'Could not get URL';
      });
  }

  // Hook into existing openMode and goBack
  var _origOpen = window.openMode;
  var _origBack = window.goBack;

  window.openMode = function (mode) {
    if (_origOpen) _origOpen(mode);
    if (mode === 'camera') {
      setTimeout(showQR, 600);
    } else {
      removeQR();
    }
  };

  window.goBack = function () {
    if (_origBack) _origBack();
    removeQR();
  };

  // Also remove on stop
  var _origStop = window.hardStop;
  window.hardStop = function () {
    if (_origStop) _origStop();
    // Keep QR visible even after stop so user can scan — don't remove here
  };

})();
