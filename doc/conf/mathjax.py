"""
Helper script for HTML builds: try to download MathJax and use its local copy
(needed for rendering formulas in offline HTML documentation).
"""

if any('html' in arg for arg in sys.argv):  # any argument has substring 'html'
    # URLs for MathJax parts
    mj_url = 'https://cdn.jsdelivr.net/npm/mathjax@4/'  # main
    mjf_url = 'https://cdn.jsdelivr.net/npm/@mathjax/'  # fonts
    mj_files = [
        (mj_url, 'tex-svg.js'),  # combined component
        (mj_url, 'input/tex/extensions/boldsymbol.js'),  # for \boldsymbol
        (mjf_url, 'mathjax-newcm-font/svg/dynamic/calligraphic.js'),  # for \mathcal
        (mjf_url, 'mathjax-newcm-font/svg/dynamic/script.js'),  # for \ell
    ]
    # local relative paths in source tree
    mj_path = 'static/mathjax/'
    mj_script = mj_path + 'tex-svg.js'

    if not os.path.exists(mj_script):
        print('Downloading MathJax...')
        try:
            from urllib.request import urlretrieve
            for url, fname in mj_files:
                dest = mj_path + fname
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                urlretrieve(url + fname, dest)
        except Exception as e:
            print(e)

    if os.path.exists(mj_script):
        print('Using local MathJax script.')
        mathjax_path = 'mathjax/tex-svg.js'  # configure sphinx.ext.mathjax
        html_js_files += ['mathjax/config.js']  # MathJax local config
    else:
        print('Local MathJax not found, using default CDN script.')
