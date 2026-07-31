import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "CloudSearch", template: "%s — CloudSearch" },
  description:
    "Agentic answer engine with citation-grounded responses. Multi-source search with AI synthesis.",
  keywords: ["search", "AI", "RAG", "citations", "answer engine"],
  openGraph: {
    title: "CloudSearch",
    description: "Agentic answer engine with inline citations",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="cs-page">
        <div className="cs-bg-orbs" aria-hidden="true" />
        {children}
      </body>
    </html>
  );
}
