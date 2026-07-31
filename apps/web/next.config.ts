import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.GATEWAY_URL ?? "http://localhost:8000"}/api/:path*`,
      },
      {
        source: "/graphql",
        destination: `${process.env.GATEWAY_URL ?? "http://localhost:8000"}/graphql`,
      },
    ];
  },
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "www.google.com" },
      { protocol: "https", hostname: "*.google.com" },
      { protocol: "https", hostname: "favicon.im" },
    ],
    // Favicons from arbitrary domains use unoptimized Image component
    dangerouslyAllowSVG: false,
  },
};

export default nextConfig;
