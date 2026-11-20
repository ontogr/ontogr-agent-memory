import React, {useEffect, useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

/**
 * Chips that show or hide the sections of every release by type, such as Added or Fixed.
 * The types come from the release sections on the page, so a new type needs no change here.
 */
export default function ChangelogFilters() {
  const [types, setTypes] = useState([]);
  const [off, setOff] = useState(() => new Set());

  useEffect(() => {
    const found = new Set(
      [...document.querySelectorAll('.changelog-section')].map((s) => s.dataset.type),
    );
    setTypes([...found]);
  }, []);

  useEffect(() => {
    document.querySelectorAll('.changelog-section').forEach((section) => {
      section.hidden = off.has(section.dataset.type);
    });
    document.querySelectorAll('.changelog-jump-link').forEach((item) => {
      const id = item.querySelector('a')?.getAttribute('href')?.slice(1);
      const target = id && document.getElementById(id)?.closest('.changelog-section');
      item.hidden = Boolean(target?.hidden);
    });
    document.querySelectorAll('.changelog-release').forEach((release) => {
      // A release with no sections stays visible. Only the filters can hide a release.
      const total = release.querySelectorAll('.changelog-section').length;
      const shown = release.querySelectorAll('.changelog-section:not([hidden])').length;
      release.hidden = total > 0 && shown === 0;
    });
  }, [off]);

  if (types.length === 0) {
    return null;
  }

  const toggle = (type) => {
    const next = new Set(off);
    if (next.has(type)) {
      next.delete(type);
    } else {
      next.add(type);
    }
    setOff(next);
  };

  return (
    <div className={styles.filters} role="group" aria-label="Filter by change type">
      <span className={styles.label}>Filter</span>
      {types.map((type) => (
        <button
          key={type}
          type="button"
          className={clsx(styles.chip, !off.has(type) && styles.on)}
          aria-pressed={!off.has(type)}
          onClick={() => toggle(type)}>
          {type}
        </button>
      ))}
    </div>
  );
}
