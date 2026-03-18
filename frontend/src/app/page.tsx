// Root page — middleware redirects to /login before this renders.
// This is a fallback in case middleware is bypassed.
import { redirect } from "next/navigation";
export default function RootPage() {
  redirect("/login");
}
