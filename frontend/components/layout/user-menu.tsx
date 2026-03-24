"use client";

import { useAuthStore } from "@/hooks/use-auth";
import { useTheme } from "next-themes";
import { useRouter } from "next/navigation";
import { LogOut, Moon, Settings, Sun, User } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export function UserMenu() {
  const { user, logout } = useAuthStore();
  const { theme, setTheme } = useTheme();
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

  const isDark = theme === "dark";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button className="flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-white/10 transition-colors outline-none">
          <div className="h-8 w-8 rounded-full bg-primary/20 flex items-center justify-center text-xs font-bold text-primary">
            {initials}
          </div>
          <span className="hidden sm:block text-sm font-medium text-white">{user.username}</span>
        </button>
      </DropdownMenuTrigger>

      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuLabel className="font-normal">
          <p className="text-sm font-medium">@{user.username}</p>
        </DropdownMenuLabel>

        <DropdownMenuSeparator />

        {/* Theme toggle */}
        <DropdownMenuItem onClick={() => setTheme(isDark ? "light" : "dark")}>
          {isDark ? <Sun className="mr-2 h-4 w-4" /> : <Moon className="mr-2 h-4 w-4" />}
          {isDark ? "Light Mode" : "Dark Mode"}
        </DropdownMenuItem>

        {/* Settings */}
        <DropdownMenuItem onClick={() => router.push("/settings")}>
          <Settings className="mr-2 h-4 w-4" />
          Settings
        </DropdownMenuItem>

        <DropdownMenuSeparator />

        {/* Logout */}
        <DropdownMenuItem onClick={handleLogout} className="text-destructive focus:text-destructive">
          <LogOut className="mr-2 h-4 w-4" />
          Log Out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
