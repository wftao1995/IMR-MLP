// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT
// Use the same TeX/SVG components as the published site, without a browser/CDN.
const fs = require('node:fs');
const path = require('node:path');

async function main() {
  const formulas = JSON.parse(fs.readFileSync(0, 'utf8'));
  const mathjax = await require(path.join(process.argv[2], 'node-main.js')).init({
    loader: {load: ['input/tex', 'output/svg']},
    // Browsers paint unknown commands red; make those compilation errors here.
    tex: {packages: {'[-]': ['noundefined']}},
    startup: {typeset: false},
  });
  const cache = new Map();
  const errors = [];
  for (const formula of formulas) {
    const key = JSON.stringify([formula.tex, formula.display]);
    if (!cache.has(key)) {
      let error = null;
      try {
        const node = await mathjax.tex2svgPromise(formula.tex, {display: formula.display});
        const output = mathjax.startup.adaptor.outerHTML(node);
        const match = output.match(/data-mjx-error="([^"]*)"/);
        if (match) error = match[1];
        else if (output.includes('data-mml-node="merror"')) error = 'MathJax merror';
      } catch (failure) {
        error = failure.message;
      }
      cache.set(key, error);
    }
    if (cache.get(key)) errors.push({...formula, error: cache.get(key)});
  }
  console.log(JSON.stringify({unique: cache.size, errors}));
}

main().catch(error => {console.error(error); process.exitCode = 1;});
