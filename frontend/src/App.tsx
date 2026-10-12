import { Suspense, lazy } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import UploadPage from "./pages/UploadPage";
import ReviewPage from "./pages/ReviewPage";
import LibraryPage from "./pages/LibraryPage";
import SearchPage from "./pages/SearchPage";
import DesignsPage from "./pages/DesignsPage";

// Konva is large; load the canvas editor (and Konva) only when it is opened.
const EditorPage = lazy(() => import("./pages/EditorPage"));

export default function App() {
  return (
    <>
      <header className="topbar">
        <div className="brand">
          Visual Asset Hub<small>Capture once. Reuse everywhere.</small>
        </div>
        <nav className="nav">
          <NavLink to="/upload">上傳</NavLink>
          <NavLink to="/library">素材庫</NavLink>
          <NavLink to="/search">搜尋</NavLink>
          <NavLink to="/designs">設計</NavLink>
        </nav>
      </header>
      <Routes>
        {/* the canvas editor uses the full window; every other page sits in the centred column */}
        <Route path="/designs/:designId" element={
          <Suspense fallback={<div className="page"><span className="spinner" /> 載入編輯器…</div>}><EditorPage /></Suspense>
        } />
        <Route path="*" element={<Pages />} />
      </Routes>
    </>
  );
}

function Pages() {
  return (
    <main className="page">
      <Routes>
        <Route path="/" element={<Navigate to="/upload" replace />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/photos/:photoId" element={<ReviewPage />} />
        <Route path="/library" element={<LibraryPage />} />
        <Route path="/search" element={<SearchPage />} />
        <Route path="/designs" element={<DesignsPage />} />
        <Route path="*" element={<div className="empty">找不到這個頁面</div>} />
      </Routes>
    </main>
  );
}
