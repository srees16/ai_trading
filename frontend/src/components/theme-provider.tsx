"use client";

import { useEffect } from "react";
import { useThemeStore } from "@/lib/store";

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const theme = useThemeStore((s) => s.theme);

  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove("light", "dark");
    root.classList.add(theme);
  }, [theme]);

  return <>{children}</>;
}

/**
 * Inline script to set theme class BEFORE React hydration.
 * Prevents flash of unstyled content (FOUC) — no visibility:hidden needed.
 * Rendered in layout.tsx inside <head>.
 */
export function ThemeScript() {
  const script = `(function(){try{var t=JSON.parse(localStorage.getItem('centurion-theme'));var c=(t&&t.state&&t.state.theme)||'light';document.documentElement.classList.add(c)}catch(e){document.documentElement.classList.add('light')}})()`;
  return <script dangerouslySetInnerHTML={{ __html: script }} />;
}
