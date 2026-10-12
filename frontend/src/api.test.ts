import { describe, expect, it } from "vitest";
import { tagUrls, withSession } from "./api";

const ID = "[0-9a-f]{32}";

describe("demo session id on API URLs", () => {
  it("appends s= as the first or an extra query parameter", () => {
    expect(withSession("/api/assets/1/thumb")).toMatch(new RegExp(`^/api/assets/1/thumb\\?s=${ID}$`));
    expect(withSession("/api/exports?design_id=2")).toMatch(new RegExp(`^/api/exports\\?design_id=2&s=${ID}$`));
  });

  it("tags API url fields at any depth and nothing else", () => {
    const design = {
      thumb_url: "/api/designs/1/thumb",
      name: "/api/not-a-url-field",
      assets: { 3: { medium_url: "/api/assets/3/medium", thumb_url: "/api/assets/3/thumb" } },
      exports: [{ url: "/api/exports/9/file" }],
      source_url: "https://github.com/x/y",
      cover_url: null,
    };
    tagUrls(design);
    const tagged = new RegExp(`\\?s=${ID}$`);
    expect(design.thumb_url).toMatch(tagged);
    expect(design.assets[3].medium_url).toMatch(tagged);
    expect(design.assets[3].thumb_url).toMatch(tagged);
    expect(design.exports[0].url).toMatch(tagged);
    expect(design.name).toBe("/api/not-a-url-field");
    expect(design.source_url).toBe("https://github.com/x/y");
    expect(design.cover_url).toBeNull();
  });
});
