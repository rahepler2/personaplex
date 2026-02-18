import ReactDOM from "react-dom/client";
import {
  createBrowserRouter,
  RouterProvider,
} from "react-router-dom";
import "./index.css";
import { Queue } from "./pages/Queue/Queue";
import { Admin } from "./pages/Admin/Admin";
import { Personas } from "./pages/Admin/Personas";
import { MCPServers } from "./pages/Admin/MCPServers";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Queue />,
  },
  {
    path: "/admin",
    element: <Admin />,
    children: [
      {
        index: true,
        element: <Personas />,
      },
      {
        path: "tools",
        element: <MCPServers />,
      },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
    <RouterProvider router={router}/>
);
