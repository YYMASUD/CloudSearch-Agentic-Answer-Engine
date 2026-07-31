import { Suspense } from "react";
import SearchPageInner from "./SearchPageInner";

export const metadata = {
  title: "Search",
};

export default function SearchPage() {
  return (
    <Suspense
      fallback={
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh" }}>
          <span className="cs-spinner" aria-label="Loading…" />
        </div>
      }
    >
      <SearchPageInner />
    </Suspense>
  );
}
