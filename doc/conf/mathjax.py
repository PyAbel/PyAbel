"""
Helper script for HTML builds: try to download MathJax and use its local copy
(needed for rendering formulas in offline HTML documentation).
"""

if any('html' in arg for arg in sys.argv):  # any argument has substring 'html'
    # URL (most complete standalone script, ~2 MB)
    mj_url = 'https://cdn.jsdelivr.net/npm/mathjax@4/tex-svg.js'
    # local file name
    mj_file = 'mathjax.js'
    # relative file path in source tree
    mj_path = 'static/' + mj_file

    if not os.path.exists(mj_path):
        print('Downloading MathJax...')
        try:
            from urllib.request import urlretrieve
            urlretrieve(mj_url, mj_path)
        except Exception as e:
            print(e)

    if os.path.exists(mj_path):
        print('Using local MathJax script.')
        mathjax_path = mj_file  # configure sphinx.ext.mathjax
        # Sphinx fails to enable MathJax automatically for "single HTML" build
        if 'singlehtml' in sys.argv:
            print('  (force inclusion)')
            html_js_files += [mj_file]  # so add it manually
    else:
        print('Local MathJax not found, using default CDN script.')
