import { useQuery } from "@tanstack/react-query";
import { api } from "../api";

/** Online demo settings; `enabled` is false when the app runs locally. */
function useDemo() {
  return useQuery({ queryKey: ["demo"], queryFn: api.demo, staleTime: Infinity }).data;
}

/** Top-bar label; the full notice is on the upload page. */
export function DemoBadge() {
  const demo = useDemo();
  if (!demo?.enabled) return null;
  return (
    <div className="demo-badge">
      <span className="badge warn" title="你上傳的照片與素材只有你看得到，閒置一段時間後自動刪除">線上 Demo</span>
      {demo.source_url && <a href={demo.source_url} target="_blank" rel="noreferrer">原始碼</a>}
    </div>
  );
}

export function DemoNotice() {
  const demo = useDemo();
  if (!demo?.enabled) return null;
  return (
    <div className="notice">
      <strong>線上 Demo</strong>：素材庫已放入 {demo.seed_photos} 張示範照片。你上傳的照片與建立的素材只有你看得到，
      閒置 {demo.idle_minutes} 分鐘後自動刪除；每人最多上傳 {demo.max_uploads} 張。
      請勿上傳含個人資料的照片。主機只有 2 核心 CPU，每張照片約需 10 秒以上。
    </div>
  );
}
