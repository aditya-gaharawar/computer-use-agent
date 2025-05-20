import { Sandbox } from "@e2b/desktop";
import { SSEEvent, ActionResponse, SSEEventType, sleep } from "@/types/api";
import {
  ComputerInteractionStreamerFacade,
  ComputerInteractionStreamerFacadeStreamProps,
} from "@/lib/streaming";
import { ResolutionScaler } from "./resolution";
import { logError, logDebug, logWarning } from "../logger";
import {
  GoogleGenerativeAI,
  HarmCategory,
  HarmBlockThreshold,
  Content,
  Part,
  Tool as GoogleTool,
  FunctionDeclaration,
  GenerateContentResponse,
  ModelParams,
} from "@google/generative-ai";

const INSTRUCTIONS = `
You are Surf, a helpful assistant that can use a computer to help the user with their tasks.
You can use the computer to search the web, write code, and more.

Surf is built by E2B, which provides an open source isolated virtual computer in the cloud made for AI use cases.
This application integrates E2B's desktop sandbox with Google's Gemini API to create an AI agent that can perform tasks
on a virtual computer through natural language instructions.

The screenshots that you receive are from a running sandbox instance, allowing you to see and interact with a real
virtual computer environment in real-time.

Since you are operating in a secure, isolated sandbox micro VM, you can execute most commands and operations without
worrying about security concerns. This environment is specifically designed for AI experimentation and task execution.

The sandbox is based on Ubuntu 22.04 and comes with many pre-installed applications including:
- Firefox browser
- Visual Studio Code
- LibreOffice suite
- Python 3 with common libraries
- Terminal with standard Linux utilities
- File manager (PCManFM)
- Text editor (Gedit)
- Calculator and other basic utilities

IMPORTANT: It is okay to run terminal commands at any point without confirmation, as long as they are required to fulfill the task the user has given. You should execute commands immediately when needed to complete the user's request efficiently.

IMPORTANT: When typing commands in the terminal, ALWAYS send a KEYPRESS ENTER action immediately after typing the command to execute it. Terminal commands will not run until you press Enter.

IMPORTANT: When editing files, prefer to use Visual Studio Code (VS Code) as it provides a better editing experience with syntax highlighting, code completion, and other helpful features.

Tool Usage Instructions:
You have a tool called "computer_action" which allows you to interact with the computer.
When you need to perform an action, call this tool with the appropriate parameters.
For example, to click at coordinates (100, 200), you would call "computer_action" with {"action_type": "click", "x": 100, "y": 200, "button": "left"}.
To type "hello world", call "computer_action" with {"action_type": "type", "text": "hello world"}.
Always provide the "action_type" and any other required parameters for that action.
After each action, you will receive a new screenshot of the computer screen. Use this to decide your next action.
If you need to wait for a moment, use the "wait" action: {"action_type": "wait", "duration_ms": 500}.
`;

interface BaseAction {
  action_type: string;
}

interface ClickActionArgs extends BaseAction {
  action_type: "click";
  x: number;
  y: number;
  button: "left" | "right" | "middle";
}

interface DoubleClickActionArgs extends BaseAction {
  action_type: "double_click";
  x: number;
  y: number;
}

interface TypeActionArgs extends BaseAction {
  action_type: "type";
  text: string;
}

interface KeypressActionArgs extends BaseAction {
  action_type: "keypress";
  keys: string[]; // e.g., ["Control", "c"]
}

interface ScrollActionArgs extends BaseAction {
  action_type: "scroll";
  scroll_x?: number; // Optional horizontal scroll
  scroll_y?: number; // Optional vertical scroll
  // For compatibility with existing scroll, let's also allow direction if x/y are not given
  direction?: "up" | "down" | "left" | "right";
  amount?: number;
}

interface MoveActionArgs extends BaseAction {
  action_type: "move";
  x: number;
  y: number;
}

interface DragActionArgs extends BaseAction {
  action_type: "drag";
  path: { x: number; y: number }[]; // Should be an array of two points
}

interface ScreenshotActionArgs extends BaseAction {
    action_type: "screenshot"; // Explicit request for screenshot
}

interface WaitActionArgs extends BaseAction {
    action_type: "wait";
    duration_ms: number; // Duration in milliseconds
}


type ComputerActionArgs =
  | ClickActionArgs
  | DoubleClickActionArgs
  | TypeActionArgs
  | KeypressActionArgs
  | ScrollActionArgs
  | MoveActionArgs
  | DragActionArgs
  | ScreenshotActionArgs
  | WaitActionArgs;


export class GoogleComputerStreamer implements ComputerInteractionStreamerFacade {
  public instructions: string;
  public desktop: Sandbox;
  public resolutionScaler: ResolutionScaler;
  private generativeAi: GoogleGenerativeAI;
  private model: ModelParams = { model: "gemini-1.5-flash-latest" }; 

  constructor(desktop: Sandbox, resolutionScaler: ResolutionScaler) {
    if (!process.env.GOOGLE_API_KEY) {
      throw new Error("GOOGLE_API_KEY is not set");
    }
    this.desktop = desktop;
    this.resolutionScaler = resolutionScaler;
    this.generativeAi = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
    this.instructions = INSTRUCTIONS;
  }

  private getTools(): GoogleTool[] {
    // Define tools for Gemini based on its tool calling capabilities
    const tools: GoogleTool[] = [
      {
        functionDeclarations: [
          {
            name: "computer_action",
            description: "Perform an action on the virtual computer. You will get a new screenshot after the action is performed.",
            parameters: {
              type: "OBJECT",
              properties: {
                action_type: {
                  type: "STRING",
                  description: "The type of action: click, double_click, type, keypress, scroll, move, drag, screenshot, wait",
                },
                x: { type: "NUMBER", description: "x-coordinate for mouse actions (click, double_click, move)", nullable: true },
                y: { type: "NUMBER", description: "y-coordinate for mouse actions (click, double_click, move)", nullable: true },
                button: { type: "STRING", description: "Mouse button (left, right, middle) for click action", nullable: true },
                text: { type: "STRING", description: "Text to type for type action", nullable: true },
                keys: {
                  type: "ARRAY",
                  items: { type: "STRING" },
                  description: "Keys to press (e.g., ['Control', 'c']) for keypress action. For single keys like Enter, use ['Enter'].",
                  nullable: true
                },
                scroll_x: { type: "NUMBER", description: "Horizontal scroll amount for scroll action", nullable: true },
                scroll_y: { type: "NUMBER", description: "Vertical scroll amount for scroll action", nullable: true },
                direction: { type: "STRING", description: "Scroll direction ('up', 'down', 'left', 'right')", nullable: true },
                amount: { type: "NUMBER", description: "Scroll amount (positive integer)", nullable: true },
                path: {
                  type: "ARRAY",
                  items: { type: "OBJECT", properties: { x: { type: "NUMBER" }, y: { type: "NUMBER" }}},
                  description: "Path for drag action, array of two points: [{x,y}, {x,y}]",
                  nullable: true
                },
                duration_ms: { type: "NUMBER", description: "Duration in milliseconds for wait action", nullable: true },
              },
              required: ["action_type"],
            },
          },
        ],
      },
    ];
    return tools;
  }

  async *stream(
    props: ComputerInteractionStreamerFacadeStreamProps
  ): AsyncGenerator<SSEEvent> {
    const { messages, signal } = props;

    const generationConfig = {
        // temperature: 0.9, 
        // topK: 1,
        // topP: 1,
        // maxOutputTokens: 2048, 
    };

    const safetySettings = [
        { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
        { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
        { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
        { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    ];

    const history: Content[] = [];
    for (const msg of messages) {
        const parts: Part[] = [];
        if (typeof msg.content === 'string') {
            parts.push({ text: msg.content });
        } else if (Array.isArray(msg.content)) { 
            msg.content.forEach(item => {
                if (typeof item === 'string') {
                    parts.push({ text: item });
                } else if (item.type === 'image_url' && item.image_url) { 
                     const base64Data = item.image_url.url.split(',')[1]; 
                     if (base64Data) {
                        parts.push({
                            inlineData: {
                                mimeType: 'image/png', 
                                data: base64Data
                            }
                        });
                     }
                }
            });
        }

        if (msg.role === "user") {
            history.push({ role: "user", parts });
        } else if (msg.role === "assistant") {
            history.push({ role: "model", parts });
        }
    }


    const modelInstance = this.generativeAi.getGenerativeModel({
        ...this.model,
        systemInstruction: { role: "system", parts: [{text: this.instructions}] }, 
        tools: this.getTools(),
        generationConfig,
        safetySettings,
    });

    const lastMessageContent = history.pop()?.parts || [{text: "What can you do?"}];


    const chat = modelInstance.startChat({
        history: history,
    });

    logDebug("Google Gemini Request:", {
        history: JSON.stringify(history, null, 2).substring(0,1000), // Truncate long histories
        lastMessage: JSON.stringify(lastMessageContent, null, 2),
        tools: this.getTools().flatMap(t => t.functionDeclarations?.map(fd => fd.name))
    });


    try {
      let streamResp = await chat.sendMessageStream(lastMessageContent);

      for await (const chunk of streamResp.stream) {
        if (signal.aborted) {
          logDebug("Stream aborted by user signal.");
          yield { type: SSEEventType.DONE, content: "Generation stopped by user" };
          return;
        }
        
        const text = chunk.text?.();
        if (text) {
          logDebug("Gemini reasoning text:", text);
          yield { type: SSEEventType.REASONING, content: text };
        }

        const functionCalls = chunk.functionCalls?.();
        if (functionCalls && functionCalls.length > 0) {
          logDebug("Gemini function calls:", JSON.stringify(functionCalls, null, 2));

          const fnCall = functionCalls[0]; 

          // Ensure action_type is part of the main action object for the frontend/client
          const actionPayload: any = { ...fnCall.args, action_type: fnCall.name };
          yield {
            type: SSEEventType.ACTION,
            action: actionPayload,
          };
          
          // Prepare actionArgs for executeAction, ensuring action_type is correctly assigned
          const actionArgs = { ...fnCall.args, action_type: fnCall.name } as ComputerActionArgs;

          const actionResponse = await this.executeAction(actionArgs);
          logDebug("Action executed, response:", actionResponse);

          yield { type: SSEEventType.ACTION_COMPLETED };

          const newScreenshotData = await this.resolutionScaler.takeScreenshot();
          const newScreenshotBase64 = Buffer.from(newScreenshotData).toString("base64");

          const toolResponsePart: Part = {
            functionResponse: {
              name: fnCall.name, 
              response: {
                 name: fnCall.name, 
                 content: { result: actionResponse?.result || "Action executed." , error: actionResponse?.error }, 
              },
            },
          };
          
          const userMessagePartsAfterToolUse: Part[] = [
            toolResponsePart,
            {
              inlineData: {
                mimeType: "image/png",
                data: newScreenshotBase64,
              },
            },
            // { text: "Screenshot after action. What's next?" } 
          ];
          
          logDebug("Sending tool response and new screenshot to Gemini:", JSON.stringify(userMessagePartsAfterToolUse, null, 2).substring(0, 500));
          streamResp = await chat.sendMessageStream(userMessagePartsAfterToolUse);
        }
      }
      logDebug("Gemini stream finished.");
      yield { type: SSEEventType.DONE };

    } catch (error: any) {
      logError("GOOGLE_STREAMER Error", error);
      const errorMessage = error.message || "An error occurred with the Google AI service. Please try again.";
      yield { type: SSEEventType.ERROR, content: errorMessage };
      yield { type: SSEEventType.DONE }; 
    }
  }

 async executeAction(
    actionArgs: ComputerActionArgs
  ): Promise<ActionResponse | void> {
    const desktop = this.desktop;
    const currentActionType = actionArgs.action_type; // action_type is now part of ComputerActionArgs
    logDebug("Executing Google AI action:", currentActionType, actionArgs);

    try {
      switch (currentActionType) { 
        case "screenshot": {
          const screenshotData = await this.resolutionScaler.takeScreenshot();
          const screenshotBase64 = Buffer.from(screenshotData).toString("base64");
          return { result: "Screenshot taken.", screenshot: screenshotBase64 };
        }
        case "double_click": {
          const coords = actionArgs as DoubleClickActionArgs;
          const coordinate = this.resolutionScaler.scaleToOriginalSpace([coords.x, coords.y]);
          await desktop.doubleClick(coordinate[0], coordinate[1]);
          break;
        }
        case "click": {
          const coords = actionArgs as ClickActionArgs;
          const coordinate = this.resolutionScaler.scaleToOriginalSpace([coords.x, coords.y]);
          if (coords.button === "left") await desktop.leftClick(coordinate[0], coordinate[1]);
          else if (coords.button === "right") await desktop.rightClick(coordinate[0], coordinate[1]);
          else if (coords.button === "middle") await desktop.middleClick(coordinate[0], coordinate[1]);
          break;
        }
        case "type": {
          await desktop.write((actionArgs as TypeActionArgs).text);
          break;
        }
        case "keypress": {
          await desktop.press((actionArgs as KeypressActionArgs).keys.join("+"));
          break;
        }
        case "move": {
          const coords = actionArgs as MoveActionArgs;
          const coordinate = this.resolutionScaler.scaleToOriginalSpace([coords.x, coords.y]);
          await desktop.moveMouse(coordinate[0], coordinate[1]);
          break;
        }
        case "scroll": {
            const scrollArgs = actionArgs as ScrollActionArgs;
            if (typeof scrollArgs.scroll_y === 'number') {
                if (scrollArgs.scroll_y < 0) await desktop.scroll("up", Math.abs(scrollArgs.scroll_y));
                else if (scrollArgs.scroll_y > 0) await desktop.scroll("down", scrollArgs.scroll_y);
            } else if (typeof scrollArgs.scroll_x === 'number') {
                logWarning("Horizontal scroll (scroll_x) not fully implemented for desktop.scroll yet with generic amounts. Use direction/amount for horizontal.");
            } else if (scrollArgs.direction && typeof scrollArgs.amount === 'number') {
                 await desktop.scroll(scrollArgs.direction, scrollArgs.amount);
            } else {
                logWarning("Scroll action requires scroll_x, scroll_y, or direction and amount.");
            }
            break;
        }
        case "drag": {
          const dragArgs = actionArgs as DragActionArgs;
          if (dragArgs.path && dragArgs.path.length === 2) {
            const startCoordinate = this.resolutionScaler.scaleToOriginalSpace([dragArgs.path[0].x, dragArgs.path[0].y]);
            const endCoordinate = this.resolutionScaler.scaleToOriginalSpace([dragArgs.path[1].x, dragArgs.path[1].y]);
            await desktop.drag(startCoordinate, endCoordinate);
          } else {
            logWarning("Drag action requires a path with exactly two points.");
            throw new Error("Drag action requires a path with exactly two points.");
          }
          break;
        }
        case "wait": {
            const waitArgs = actionArgs as WaitActionArgs;
            if (waitArgs.duration_ms > 0) {
                await sleep(waitArgs.duration_ms);
            } else {
                logWarning("Wait action requires a positive duration_ms.");
            }
            break;
        }
        default: {
          const unknownAction = actionArgs as any;
          logWarning("Unknown action type received:", currentActionType, unknownAction);
          throw new Error(`Unknown action type: ${currentActionType}`);
        }
      }
      return { result: `Action ${currentActionType} executed successfully.` };
    } catch (error: any) {
      logError(`Error executing action ${currentActionType}:`, error);
      return { result: `Error executing action ${currentActionType}: ${error.message}`, error: true };
    }
  }
}
