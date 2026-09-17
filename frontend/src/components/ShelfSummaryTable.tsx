import type { ShelfRegion } from "../types/analysis";
import { StatusBadge } from "./StatusBadge";

export function ShelfSummaryTable({ regions }: { regions: ShelfRegion[] }) {
  if (regions.length === 0) {
    return <p className="empty-state">No shelf regions were detected in this image.</p>;
  }

  return (
    <table className="summary-table">
      <thead>
        <tr>
          <th>Shelf</th>
          <th>Facings</th>
          <th>Occupancy</th>
          <th>Gaps</th>
          <th>Est. missing</th>
          <th>Confidence</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {regions.map((region) => (
          <tr key={region.region_id}>
            <td>Shelf {region.region_id + 1}</td>
            <td>{region.facing_count}</td>
            <td>{region.status === "unknown" ? "—" : `${(region.occupancy_ratio * 100).toFixed(0)}%`}</td>
            <td>{region.gaps.length}</td>
            <td>{region.gaps.reduce((sum, gap) => sum + gap.estimated_missing_facings, 0)}</td>
            <td>{(region.avg_confidence * 100).toFixed(0)}%</td>
            <td>
              <StatusBadge status={region.status} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
