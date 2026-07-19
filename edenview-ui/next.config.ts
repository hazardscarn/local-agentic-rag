import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The dev-only route indicator collides with the sidebar's own status footer
  // (same bottom-left corner) -- purely a dev-mode overlay, no production effect.
  devIndicators: false,
};

export default nextConfig;
