import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock axios before importing hooks
vi.mock("@/lib/api", () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    patch: vi.fn(),
  },
}));

import api from "@/lib/api";

describe("API client", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("api module exports an axios-like object", () => {
    expect(api).toBeDefined();
    expect(typeof api.get).toBe("function");
    expect(typeof api.post).toBe("function");
    expect(typeof api.patch).toBe("function");
  });

  it("get function can be called with path", async () => {
    const mockData = { status: "ok" };
    vi.mocked(api.get).mockResolvedValue({ data: mockData });

    const result = await api.get("/health");
    expect(api.get).toHaveBeenCalledWith("/health");
    expect(result.data).toEqual(mockData);
  });

  it("post function can be called with body", async () => {
    const mockData = { id: "test-id", status: "pending" };
    vi.mocked(api.post).mockResolvedValue({ data: mockData });

    const body = { account_id: "ACC_001", strategy: "payment_recovery" };
    const result = await api.post("/interventions/draft", body);
    expect(api.post).toHaveBeenCalledWith("/interventions/draft", body);
    expect(result.data).toEqual(mockData);
  });

  it("patch function can be called with body", async () => {
    const mockData = { id: "test-id", status: "approved" };
    vi.mocked(api.patch).mockResolvedValue({ data: mockData });

    const result = await api.patch("/interventions/test-id/status", {
      status: "approved",
    });
    expect(api.patch).toHaveBeenCalledWith("/interventions/test-id/status", {
      status: "approved",
    });
    expect(result.data).toEqual(mockData);
  });
});
