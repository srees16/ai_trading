/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  async rewrites() {
    // In production (Vercel), NEXT_PUBLIC_API_URL is set and the browser calls
    // the backend directly — no rewrite proxy needed. Vercel's edge proxy has
    // strict timeouts that cause ROUTER_EXTERNAL_TARGET_ERROR with slow backends.
    // The rewrite is only used for local dev (localhost:9001 fallback).
    if (process.env.NEXT_PUBLIC_API_URL) {
      return [];
    }
    return [
      {
        source: "/api/v1/:path*",
        destination: "http://localhost:9001/api/v1/:path*",
      },
    ];
  },
};

module.exports = nextConfig;
