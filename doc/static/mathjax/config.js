MathJax = {
  loader: {
    paths: {
      /* Configure local MathJax to use local fonts.
         Relative URLs are interpreted w.r.t. current HTML file, so will be
         different at different tree levels, thus the correct URL within
         "static/mathjax/" must be created from this script's URL. */
      'mathjax-newcm': document.currentScript.src.replace(/[^\/]+$/, 'mathjax-newcm-font')
    }
  }
}
