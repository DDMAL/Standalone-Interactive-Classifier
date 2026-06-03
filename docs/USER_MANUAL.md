# Interactive Classifier — User Manual

The **Interactive Classifier (IC)** is a web tool for classifying *neumes* (medieval
chant notation) on a manuscript page. You upload a page image and its bounding-box
annotations, then interactively label each glyph. A k-Nearest-Neighbors (kNN) model
re-classifies the remaining glyphs as you confirm labels, so accuracy improves the
more you correct. When finished, you export the result as GameraXML.

---

## 1. Getting started

### Upload screen

| Field | What to provide |
|-------|-----------------|
| **Page image** | The manuscript page (TIFF / PNG / JPG). |
| **Annotations file** | The bounding boxes for the page. |
| **Annotation format** | `MOTHRA JSON` or `YOLO TXT` — must match your annotations file. |
| **Training set** *(optional)* | A pre-trained kNN model to seed automatic classification. |
| **Vocabulary** *(optional)* | A list of class names used for autocomplete. The **Available classes** box shows what the chosen vocabulary contains. |

Click **Start session** to load the page and begin.

### The session layout

Once a session starts you see four regions:

- **Left — Class tree:** every class name in use, as a collapsible hierarchy. Click a
  class to select all of its glyphs.
- **Center — Page view:** the manuscript image with bounding-box overlays.
- **Right — Glyph grid:** thumbnails of all glyphs, sorted by confidence (least
  confident first, so your attention goes where it's needed).
- **Right sidebar — Edit panel:** appears when one or more glyphs are selected.

### Color coding

| Glyph state | Appearance |
|-------------|-----------|
| Manually labeled | **Green** box |
| Auto-classified (kNN), unselected | Faint / transparent box |
| Auto-classified, selected | **Blue** box |
| Hovered | **Amber** highlight |
| Text / Staves glyphs | Shown only when selected or hovered (dashed box) |

---

## 2. Keyboard shortcuts

> Shortcuts that act on the page (zoom, pan, clear) are ignored while you are typing
> in a text field, dropdown, or rename box.

### Page navigation

| Key | Action |
|-----|--------|
| `+` or `=` | Zoom in (1.2×) |
| `-` or `_` | Zoom out |
| `0` | Reset zoom and pan |
| `←` `→` `↑` `↓` | Pan the page by 40 px (matches scroll-drag direction) |
| `Esc` | Clear the current selection |

### Editing (Edit panel focused)

| Key | Action |
|-----|--------|
| `Enter` | Apply the typed class name and reclassify the selected glyph(s) |
| `Cmd`/`Ctrl` + `G` | Group the selected glyphs (2 or more selected) |
| `Cmd`/`Ctrl` + `E` | Jump focus to the class-name input |

### Class-name autocomplete (while typing a class name)

| Key | Action |
|-----|--------|
| `↓` / `↑` | Move through the suggestion list (wraps around) |
| `Enter` | Accept the highlighted suggestion |
| `Esc` | Close the suggestion list |

### Rename box (in the class tree)

| Key | Action |
|-----|--------|
| `Enter` | Save the new name |
| `Esc` | Cancel the rename |

---

## 3. Mouse operations

### Selecting glyphs

| Action | Result |
|--------|--------|
| **Click** a bounding box or grid tile | Select that single glyph |
| **Shift / Cmd / Ctrl + Click** | Toggle that glyph in/out of a multi-selection |
| **Drag** on empty page area | Marquee (rubber-band) select everything inside the rectangle |
| **Shift / Cmd / Ctrl + Drag** | Marquee that *adds* to the current selection |
| **Click** empty page area | Clear the selection |
| **Click** a class in the left tree | Select all glyphs of that class (including sub-classes) |
| **Hover** a box or tile | Highlight it for visual feedback |

Clicking a tile in the grid also scrolls/centers the page on that glyph if it's
currently off-screen.

### Zoom & pan with the mouse

| Action | Result |
|--------|--------|
| **Scroll wheel** | Pan the page |
| **Ctrl / Cmd + Scroll** (or trackpad pinch) | Zoom, anchored at the cursor |

---

## 4. Core tasks

### Classify a single glyph
1. Click the glyph (on the page or in the grid).
2. The Edit panel shows its image, current class, kNN confidence, source
   (Manual / Auto), position, and size.
3. Type or pick a **class name**.
4. Press `Enter` or click **Apply & reclassify**. The label is saved and the kNN
   model re-runs over the remaining auto glyphs.

### Classify many glyphs at once
1. Multi-select glyphs (Shift/Cmd-click or marquee drag).
2. The panel summarizes how many are Neumes vs. non-Neumes.
3. Type a class name and click **Apply to N Neumes** (or press `Enter`).
   Non-Neume glyphs in the selection are skipped automatically.

### Move a glyph to a different category
In the single-glyph Edit panel, click **→ Neumes**, **→ Text**, or **→ Staves**
to reassign its MOTHRA category. Only the categories it isn't currently in are shown.

### Split a glyph
1. Select the glyph and click **Split glyph…**.
2. In the dialog, **drag** on the image to draw a rectangle for each piece (shown as a
   dashed green outline). Rectangles snap to the pixel grid and clip to the glyph's bounds.
3. Use the **×** on a rectangle to remove it, or **Clear all** to start over.
4. Click **Split into N glyphs** to create that many new, unclassified children
   (or **Cancel**).

### Group glyphs into one
1. Multi-select 2 or more Neumes.
2. Press `Cmd`/`Ctrl` + `G` or click **Group as new glyph**.
3. Enter a class name and click **Group**. The selected glyphs merge into a single
   new manual glyph.

### Delete and restore
- Click **Delete glyph** (single) or **Delete N glyphs** (multi) to *soft-delete*.
  Deleted glyphs move to the **Deleted** section at the bottom of the grid.
- Expand **Deleted** and click **Put back** to restore a glyph to its original category.
- Soft-deleted glyphs are only permanently removed when you export.

### Manage classes (left tree)
Hover a class node to reveal its actions:
- **Select** — select all glyphs in that class.
- **Rename** — edit the class-name segment inline (`Enter` to save, `Esc` to cancel).
- **Delete** — remove the class (with confirmation).

Use the chevron (`<` / `>`) at the edge of the tree panel to collapse it to a thin
strip or expand it again.

---

## 5. Toolbar & finishing

| Control | Action |
|---------|--------|
| **k = 1 / 3 / 5 / 7** | Number of neighbors the kNN model uses when classifying. Default is 3. |
| **New session** | Discard the current session and return to the upload screen. |
| **Save** | Persist the current session state to the backend. |
| **Complete & Export** | Finalize the session: commit deletions and write the output as GameraXML. |

---

## 6. Quick reference card

```
ZOOM / PAN          SELECTION                 EDITING
  +  /  =  zoom in     click      select one     Enter   apply + reclassify
  -  /  _  zoom out    Shift+clk  toggle          Cmd/Ctrl+G  group
  0        reset       drag       marquee         Cmd/Ctrl+E  focus name field
  arrows   pan 40px    Shift+drag add-marquee     autocomplete: ↑/↓ Enter Esc
  wheel    pan         click bg   clear
  Ctrl+wheel zoom      Esc        clear
```
