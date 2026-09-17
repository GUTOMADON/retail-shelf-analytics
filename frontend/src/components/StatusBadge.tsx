import type { RegionStatus } from "../types/analysis";

const STATUS_LABEL: Record<RegionStatus, string> = {
  ok: "OK",
  understocked: "Under-stocked",
  empty: "Empty",
};

export function StatusBadge({ status }: { status: RegionStatus }) {
  return <span className={`status-badge status-badge--${status}`}>{STATUS_LABEL[status]}</span>;
}
