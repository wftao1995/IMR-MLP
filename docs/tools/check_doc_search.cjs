// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT
// Execute the generated search language data, without a browser or network.
// This catches missing stemmer globals that an HTML link check cannot detect.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

for (const directory of process.argv.slice(2)) {
  const context = vm.createContext({ window: {} });
  vm.runInContext(fs.readFileSync(path.join(directory, '_static/language_data.js'), 'utf8'), context);
  const Stemmer = context.Stemmer || context.window.Stemmer;
  assert.equal(typeof Stemmer, 'function', `${directory}: missing search stemmer`);
  const stemmer = new Stemmer();
  let index;
  const receiver = vm.createContext({ Search: { setIndex: value => { index = value; } } });
  vm.runInContext(fs.readFileSync(path.join(directory, 'searchindex.js'), 'utf8'), receiver);
  for (const query of path.basename(directory) === 'zh' ? ['force_weight', '训练'] : ['force_weight']) {
    const term = stemmer.stemWord(query);
    const documents = [index.terms[term], index.titleterms[term]]
      .flatMap(matches => matches === undefined ? [] : Array.isArray(matches) ? matches : [matches]);
    assert(documents.length, `${directory}: query ${query} missing from search index`);
    assert(documents.some(id => index.docnames[id] === 'guides/workflows'), `${query} must find training`);
  }
  console.log(`${directory}: generated search script, training queries OK`);
}
