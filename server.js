import express from "express";
import multer from "multer";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const app = express();
const PORT = 3000;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY
});

const upload = multer({
    storage: multer.memoryStorage(),
    limits: {
        fileSize: 10 * 1024 * 1024
    }
});

app.use(express.static(__dirname));

app.post("/analyze", upload.single("foodImage"), async (req, res) => {

    try {
        if (!req.file) {
            return res.status(400).json({
                error: "No image was uploaded."
            });
        }

        const base64Image = req.file.buffer.toString("base64");

        const prompt = `
You are FoodWise AI, an intelligent food recognition and recipe assistant.

Look carefully at the uploaded food image.

Identify the most likely food or dish.

Then provide:
1. The detected food name
2. A short description
3. Three suitable recipe suggestions
4. Ingredients for each recipe
5. Preparation time
6. Cooking time
7. Clear step-by-step cooking instructions

Do not pretend to know something with certainty if the image is unclear.
When uncertain, say "Possible match" and give your best guess.

Return ONLY valid JSON using this structure:

{
  "food_name": "string",
  "confidence": "string",
  "description": "string",
  "recipes": [
    {
      "name": "string",
      "prep_time": "string",
      "cook_time": "string",
      "ingredients": ["string"],
      "steps": ["string"]
    }
  ]
}
`;

        const interaction = await ai.interactions.create({
            model: "gemini-3.8-flash",
            input: [
                {
                    type: "image",
                    data: base64Image,
                    mime_type: req.file.mimetype
                },
                {
                    type: "text",
                    text: prompt
                }
            ],
            response_format: {
                type: "text",
                mime_type: "application/json",
                schema: {
                    type: "object",
                    properties: {
                        food_name: {
                            type: "string"
                        },
                        confidence: {
                            type: "string"
                        },
                        description: {
                            type: "string"
                        },
                        recipes: {
                            type: "array",
                            items: {
                                type: "object",
                                properties: {
                                    name: {
                                        type: "string"
                                    },
                                    prep_time: {
                                        type: "string"
                                    },
                                    cook_time: {
                                        type: "string"
                                    },
                                    ingredients: {
                                        type: "array",
                                        items: {
                                            type: "string"
                                        }
                                    },
                                    steps: {
                                        type: "array",
                                        items: {
                                            type: "string"
                                        }
                                    }
                                },
                                required: [
                                    "name",
                                    "prep_time",
                                    "cook_time",
                                    "ingredients",
                                    "steps"
                                ]
                            }
                        }
                    },
                    required: [
                        "food_name",
                        "confidence",
                        "description",
                        "recipes"
                    ]
                }
            }
        });

        const result = JSON.parse(interaction.output_text);

        res.json(result);

    } catch (error) {

        console.error("AI Error:", error);

        res.status(500).json({
            error: "Could not analyze the image. Please check your API key and try again."
        });
    }
});

app.listen(PORT, () => {
    console.log(`FoodWise AI running at http://localhost:${PORT}`);
});