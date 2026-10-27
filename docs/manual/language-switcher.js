// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT
document.addEventListener('click', (event) => {
  for (const menu of document.querySelectorAll('.imr-language-switch[open]')) {
    if (!menu.contains(event.target)) menu.open = false;
  }
});

document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape') return;
  for (const menu of document.querySelectorAll('.imr-language-switch[open]')) {
    const hadFocus = menu.contains(document.activeElement);
    menu.open = false;
    if (hadFocus) menu.querySelector('summary').focus();
  }
});
