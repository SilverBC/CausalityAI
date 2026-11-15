/* Icons from rebuild-plan/assets/icons, inlined so they use currentColor and follow the theme. */

const Svg = ({ size = 18, children, ...rest }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} aria-hidden="true" focusable="false" {...rest}>{children}</svg>
);

export const SunIcon = (p) => (
  <Svg {...p} fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <circle cx="12" cy="12" r="4.2" />
    <path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5.3 5.3l1.6 1.6M17.1 17.1l1.6 1.6M5.3 18.7l1.6-1.6M17.1 6.9l1.6-1.6" />
  </Svg>
);

export const MoonIcon = (p) => (
  <Svg {...p} fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round">
    <path d="M20.2 14.6A8.5 8.5 0 0 1 9.4 3.8a8.5 8.5 0 1 0 10.8 10.8z" />
  </Svg>
);

export const SidebarIcon = (p) => (
  <Svg size={20} {...p} fill="none" stroke="currentColor" strokeWidth="1.8">
    <rect x="3" y="4" width="18" height="16" rx="2.5" /><path d="M9 4v16" />
  </Svg>
);

export const UploadIcon = (p) => (
  <Svg size={28} {...p}>
    <path d="M12 16V4m0 0l-4 4m4-4l4 4M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3" fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
  </Svg>
);

/** Treatment dot -> outcome dot. Colours come from the theme tokens. */
export const Logo = ({ size = 26 }) => (
  <svg viewBox="0 0 32 32" width={size} height={size} aria-hidden="true" focusable="false">
    <circle cx="7" cy="16" r="5" fill="var(--treat)" />
    <circle cx="25" cy="16" r="5" fill="var(--ctrl)" />
    <path d="M12 16h8" stroke="var(--ink2)" strokeWidth="2.5" />
    <path d="M17 12l4 4-4 4" fill="none" stroke="var(--ink2)" strokeWidth="2.5" />
  </svg>
);

export const TrashIcon = (p) => (
  <Svg size={16} {...p} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
    <path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3" />
  </Svg>
);

export const WarnIcon = (p) => (
  <Svg size={16} {...p} fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3.5 2.5 20h19z" /><path d="M12 10v4.5M12 17.3v.2" />
  </Svg>
);

export const CheckIcon = (p) => (
  <Svg size={15} {...p} fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
    <path d="M5 12.5l4.5 4.5L19 7.5" />
  </Svg>
);

export const ChevronIcon = (p) => (
  <Svg size={16} {...p} fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M6 9l6 6 6-6" />
  </Svg>
);

export const GraphIcon = (p) => (
  <Svg size={20} {...p}>
    <circle cx="5" cy="6" r="2.6" fill="currentColor" /><circle cx="19" cy="6" r="2.6" fill="currentColor" />
    <circle cx="12" cy="18" r="2.6" fill="currentColor" />
    <path d="M7 7l4 9M17 7l-4 9M7.6 6h8.8" stroke="currentColor" strokeWidth="1.6" fill="none" />
  </Svg>
);

/** The Investigator: a line-art alligator head (thick rounded strokes, open jaws, zigzag teeth, eye, nostril) in a
    deerstalker tipped ~45° back. Strokes use currentColor; the hat's crown is filled with --accent (the button colour)
    so it covers the lines behind it. From src/assets/mascot/gator-current-v5.svg; earlier versions are kept there. */
export const GatorIcon = ({ size = 52 }) => (
  <svg viewBox="-3 -4 52 52" width={size} height={size} aria-hidden="true" focusable="false" fill="none" stroke="currentColor"
    strokeWidth="2.7" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20.6 18.8C24 14.6 29.2 12.6 31 9.8C30.4 6.4 32.6 3.4 35.6 3.2 38.8 3 41.2 6.4 40.8 9.6c-.2 1.6-1 2.8-2.6 4.2 L38.2 13.8 L35.0 13.1 L34.5 16.4 L31.2 15.7 L30.7 19.0 L27.5 18.3 L26.9 21.6 L23.7 20.9 L23.2 24.2 C20.6 27.8 17.4 31 13.8 31.4" />
    <path d="M13.8 31.4c4.4 1.4 9.4.6 12.6-3.6 L26.4 27.8 L29.2 29.0 L29.7 25.9 L32.5 27.1 L33.0 24.1 L35.8 25.3 L36.3 22.2 L39.1 23.4 L39.6 20.4 L39.8 19c3 .2 4.6 1.6 4.9 3.8.5 3.6-2 6.8-6 8.6-5 2.2-9.8 4-12.2 7.2-2.2 3.2-6.4 5-10.4 4.8-4.8-.2-7.2-3.8-7.4-10.4" />
    <path d="M7.6 30.4c-.1-1.6 0-3 .4-4.4" />
    <path d="M16.8 18.6c1.4-.2 2.6 0 3.8.4" />
    <ellipse cx="35.8" cy="7.3" rx="1" ry="1.5" strokeWidth="1.9" />
    <ellipse cx="14.8" cy="24.9" rx="2.7" ry="1.95" />
    <circle cx="14" cy="25.1" r=".9" fill="currentColor" stroke="none" />
    <g transform="translate(11.4 20.8) rotate(-45.0) scale(1.25) translate(-14.8 -18.4)" strokeWidth="2.16">
      <path d="M10.6 17.6C10.2 13.2 12.8 10.6 15.8 10.8c3 .2 5 2.6 4.4 6.6z" fill="var(--accent)" />
      <path d="M7.4 19.4C12 17.2 17.4 17 22.2 18.8" />
      <path d="M11.2 14.8c2.8-.9 5.6-.9 8.8.3" strokeWidth="1.20" />
    </g>
  </svg>
);
