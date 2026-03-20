/** @type {import('next').NextConfig} */
const nextConfig = {
  // 'standalone' for Docker; Vercel ignores this and uses its own build system
  ...(process.env.NODE_ENV === "production" && !process.env.VERCEL ? { output: "standalone" } : {}),
  reactStrictMode: false,
  experimental: {
    optimizePackageImports: [
      "lucide-react",
      "recharts",
      "date-fns",
      "@tanstack/react-table",
      "sonner",
      "@radix-ui/react-accordion",
      "@radix-ui/react-avatar",
      "@radix-ui/react-checkbox",
      "@radix-ui/react-dialog",
      "@radix-ui/react-dropdown-menu",
      "@radix-ui/react-label",
      "@radix-ui/react-progress",
      "@radix-ui/react-scroll-area",
      "@radix-ui/react-select",
      "@radix-ui/react-separator",
      "@radix-ui/react-slider",
      "@radix-ui/react-slot",
      "@radix-ui/react-switch",
      "@radix-ui/react-tabs",
      "@radix-ui/react-tooltip",
    ],
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:9001"}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
