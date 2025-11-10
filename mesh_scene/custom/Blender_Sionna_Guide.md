# Setting Up Blender with Mitsuba Add-on for Sionna

## Prerequisites

**Recommended Setup:**

- Blender 3.6.9 (version 4.0 has known compatibility issues with Sionna)
- Your Blender 3.9 worked, but 3.6.9 is more stable for this workflow
- The add-on requires Blender 2.93 and above to function

---

## Step 1: Download the Mitsuba Add-on

1. Go to: https://github.com/mitsuba-renderer/mitsuba-blender
2. Download the latest release from the release section
3. Download the `mitsuba-blender.zip` file (don't unzip it!)

---

## Step 2: Install the Add-on in Blender

1. Open Blender
2. Go to Edit → Preferences → Add-ons → Install
3. Select the downloaded ZIP archive
4. Search for "Mitsuba-Blender" in the add-on search bar on the top right
5. Click on the checkbox next to the add-on name to activate it

---

## Step 3: Install Dependencies

After enabling the add-on, you need to install Mitsuba dependencies:

1. Expand the Mitsuba-Blender add-on item in the preferences
2. Click "Install dependencies using pip" to download the latest package
3. **Important:** Blender will appear to hang while packages are being downloaded - do not interrupt it (this can take several minutes)
4. Once Mitsuba is correctly detected and initialized, the status message should display a check mark indicating the add-on is ready

**Alternative:** Check "Use custom Mitsuba path" and browse to your Mitsuba build directory if you have Mitsuba installed separately

---

## Step 4: Prepare Your Scene for Export

### A. Triangulate All Meshes

**This is critical** - Sionna only accepts triangle meshes!

**Method 1 - Per Object:**

```
1. Select your object
2. Press Tab (Enter Edit Mode)
3. Press A (Select All)
4. Press Ctrl+T (Triangulate)
5. Press Tab (Return to Object Mode)
```

**Method 2 - All Objects at Once:**

```
1. Select all objects (A in Object Mode)
2. Press Tab (all mesh objects enter Edit Mode)
3. Press A (Select all vertices)
4. Press Ctrl+T (Triangulate)
5. Press Tab (Return to Object Mode)
```

### B. Verify Object Types

All objects must be type "Mesh". Right-click on the object → Convert to → Mesh

---

## Step 5: Export to Mitsuba XML

1. Go to **File → Export → Mitsuba (.xml)**
2. **Critical Export Settings:**
   - Check the box "Export IDs"
   - Set axis correctly: Y forward, Z up
3. Choose your export location
4. Click **Export Mitsuba**

The export will create:

- An `.xml` file (your scene description)
- A `meshes/` folder containing `.ply` files for each object

---

## Step 6: Verify the Export

Check your export folder - you should have:

```
your_scene.xml
meshes/
  └── Cube.ply (or your object names)
```

Make sure the "meshes" folder is in the same directory as the XML file when loading in Sionna

---

## Step 7: Load in Sionna

In your Python/Jupyter notebook:

```python
from sionna.rt import load_scene

# Make sure both XML and meshes folder are in the same directory
scene = load_scene("path/to/your_scene.xml")
scene.preview()
```

---

## Important Notes

### What the Add-on Does:

The add-on is really just for exporting the XML files - the rendering inside of Blender is not finished yet

You use it to:

- ✅ Export Blender scenes to Mitsuba XML format
- ✅ Import Mitsuba scenes back into Blender for editing
- ❌ NOT for rendering in Blender itself

### Troubleshooting:

**If Mitsuba doesn't appear in Export menu:**

- Make sure the add-on is enabled (green checkbox)
- Restart Blender
- Check that dependencies installed successfully (green checkmark in preferences)

**If Sionna gives "triangle mesh" error:**

- You forgot to triangulate! Go back and use Ctrl+T on all meshes

**If Sionna can't find PLY files:**

- Make sure the `meshes` folder is in the same directory as the XML
- Check that the paths in the XML are relative (like `meshes/Cube.ply`)

---

## Quick Workflow Summary

```
1. Install add-on → Install dependencies
2. Create/import your 3D model in Blender
3. Triangulate all meshes (Ctrl+T)
4. File → Export → Mitsuba (.xml)
5. Keep XML + meshes folder together
6. Load in Sionna
```

That's it! Your successful export shows you've got everything working correctly. 🎉
