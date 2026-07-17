"use client";

import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { HelpCircle } from "lucide-react";

export function FieldHelp({ children }: { children: React.ReactNode }) {
  return (
    <Popover>
      <PopoverTrigger className="inline-flex size-4 items-center justify-center rounded-full text-muted-foreground hover:text-foreground">
        <HelpCircle className="size-3.5" />
      </PopoverTrigger>
      <PopoverContent className="text-sm text-muted-foreground">{children}</PopoverContent>
    </Popover>
  );
}
