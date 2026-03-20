"use client";

import { useAuthStore } from "@/hooks/use-auth";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { LogOut, User } from "lucide-react";
import { useRouter } from "next/navigation";

export function UserMenu() {
  const { user, logout } = useAuthStore();
  const router = useRouter();

  if (!user) return null;

  const handleLogout = async () => {
    await logout();
    router.push("/login");
  };

  const initials = user.name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);

  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-2">
        <div className="h-8 w-8 rounded-full bg-primary/20 flex items-center justify-center text-xs font-bold text-primary">
          {initials}
        </div>
        <div className="hidden sm:block">
          <p className="text-sm font-medium leading-none">{user.name}</p>
          <Badge variant="secondary" className="mt-0.5 text-[0.6rem]">
            {user.role}
          </Badge>
        </div>
      </div>
      <Button variant="ghost" size="icon" onClick={handleLogout} title="Logout">
        <LogOut className="h-4 w-4" />
      </Button>
    </div>
  );
}
