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
