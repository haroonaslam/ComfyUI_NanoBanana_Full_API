Thought it is time to share the custom node ! it is working well. You need to setup your Google Studio API key and add to the node.

Features:
- Set API key
- Set model (Stable and preview endpoints)
- Select up to 5 images. No need to stitch or batch. Just directly load via "load image" comfy node and connect to the custom node. all image inputs are optional. can directly use a prompt to generate from scratch.
- Set system instructions if you need specific styles while modifying the prompts
- select all available aspect ratios
- Candidate count (keep it to default 1. this is WIP to setup number of generations). For now, just queue required number in comfyui.
- Safety feature flags selectable via UI. It does reduce the probability of prompt being directly rejected. Sometimes you can get lucky with suggestive but not NSFW as the filters I believe work for prompt level. Default post generation filtering is at model level and not available via API.
- Debug outputs in console
- Error messages as node output - just connect the text output to a text display node.
- In-painting mode. via Edit mode (Yes/No). when this is selected to No, all 5 image options are handled if connected. When this is selected to Yes, only image1 input is enabled. This is an in-paint mode where the node expects a masked input to be connected (via same "load image" -> mask output if you just generate mask via the load image right click option). The custom node will handle the mask input to parse correctly and pass on to nanobanana API to keep things simple !
- basically just use load image node, load an image, right click and mask, connect image output to image1 input. connect mask output to mask input. Write your prompt on what you need to change. and queue !
- change temperature and Top P as required.
- Added a screenshot of node below in first comment.

You can also use image as input and in prompt as a question to describe the image, the text output is your friend in this case.
Edit 1: Just add the node Custom_nanobananav3.py to custom node directory and in comfyui search by NanoBanana to add (it should show API V2) . image posted as well :)
