"use client"

import { Autocomplete } from "@base-ui/react/autocomplete"

import { cn } from "@/lib/utils"

interface ComboboxProps {
  value: string
  onValueChange: (value: string) => void
  items: string[]
  placeholder?: string
  emptyText?: string
  id?: string
  className?: string
}

// Free-solo combobox: type a brand-new name (creates it) or pick one of the existing
// `items` from the dropdown -- built on Base UI's Autocomplete, which (unlike Select)
// treats the input's own text as the value rather than requiring a match against a
// fixed item list.
function Combobox({ value, onValueChange, items, placeholder, emptyText = "No matches -- press Enter to use this as a new name", id, className }: ComboboxProps) {
  return (
    <Autocomplete.Root items={items} value={value} onValueChange={onValueChange} openOnInputClick>
      <Autocomplete.Input
        id={id}
        placeholder={placeholder}
        className={cn(
          "h-8 w-full min-w-0 rounded-lg border border-input bg-transparent px-2.5 py-1 text-base transition-colors outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 md:text-sm dark:bg-input/30",
          className
        )}
      />
      <Autocomplete.Portal>
        <Autocomplete.Positioner className="z-50 outline-none" sideOffset={4}>
          <Autocomplete.Popup className="w-[var(--anchor-width)] max-w-[var(--available-width)] rounded-lg border border-border bg-popover text-popover-foreground shadow-md">
            <Autocomplete.Empty className="px-3 py-2 text-xs text-muted-foreground">{emptyText}</Autocomplete.Empty>
            <Autocomplete.List className="max-h-52 overflow-y-auto p-1 outline-none">
              {(item: string) => (
                <Autocomplete.Item
                  key={item}
                  value={item}
                  className="cursor-default rounded-md px-2 py-1.5 text-sm outline-none select-none data-highlighted:bg-accent data-highlighted:text-accent-foreground"
                >
                  {item}
                </Autocomplete.Item>
              )}
            </Autocomplete.List>
          </Autocomplete.Popup>
        </Autocomplete.Positioner>
      </Autocomplete.Portal>
    </Autocomplete.Root>
  )
}

export { Combobox }
