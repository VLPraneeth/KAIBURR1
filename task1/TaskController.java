package com.kaiburr.taskmanager.controller;

import com.kaiburr.taskmanager.models.Task;
import com.kaiburr.taskmanager.models.TaskExecution;
import com.kaiburr.taskmanager.repository.TaskRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.concurrent.TimeUnit;

@CrossOrigin(origins = "http://localhost:3000")  
@RestController
@RequestMapping("/tasks")  
public class TaskController {

    @Autowired
    private TaskRepository taskRepository;

    // Dangerous Commands Restriction
    private static final List<String> DANGEROUS_COMMANDS = Arrays.asList(
        "rm -rf", "shutdown", "poweroff", "reboot", "mkfs", "wget", "curl", 
        "dd", "mv /", "cp /", "echo >", "chmod 777", "chown root", 
        "killall", "kill -9", "iptables", "nano /etc/passwd", "vim /etc/shadow"
    );

    @GetMapping("/ping")
    public String ping() {
        return "Task Manager API is running!";
    }

    @GetMapping
    public List<Task> getAllTasks() {
        return taskRepository.findAll();
    }

    @GetMapping("/{taskId}")
    public Task getTaskById(@PathVariable String taskId) {
        return taskRepository.findById(taskId)
                .orElseThrow(() -> new RuntimeException("Task not found"));
    }

    @GetMapping("/search")
    public List<Task> searchTasksByName(@RequestParam String name) {
        return taskRepository.findByNameContaining(name);
    }

    // Create a Task (No execution here, just storing the command)
    @PostMapping
    public Task createTask(@RequestBody Task task) {
        if (task.getTaskExecutions() == null) {
            task.setTaskExecutions(new ArrayList<>());
        }

        return taskRepository.save(task);
    }

    // Execute a Task (Executes the stored command)
    @PostMapping("/{taskId}/execute")
    public Task executeTask(@PathVariable String taskId) {
        Task task = taskRepository.findById(taskId)
                .orElseThrow(() -> new RuntimeException("Task not found"));

        if (task.getTaskExecutions() == null) {
            task.setTaskExecutions(new ArrayList<>());
        }

        TaskExecution execution = new TaskExecution();
        execution.setStartTime(new Date());

        // Check for unsafe commands
        String output;
        if (isDangerousCommand(task.getCommand())) {
            output = "Unsafe command detected: " + task.getCommand();
        } else {
            output = executeShellCommand(task.getCommand());
            if (output == null || output.trim().isEmpty()) {
                output = "Command executed successfully (No output)";
            }
        }

        execution.setOutput(output);
        execution.setEndTime(new Date());
        task.getTaskExecutions().add(execution);

        return taskRepository.save(task);
    }

    @PutMapping("/{taskId}")
    public Task updateTask(@PathVariable String taskId, @RequestBody Task updatedTask) {
        Optional<Task> optionalTask = taskRepository.findById(taskId);
        if (optionalTask.isPresent()) {
            Task existingTask = optionalTask.get();
            existingTask.setName(updatedTask.getName());
            existingTask.setOwner(updatedTask.getOwner());
            existingTask.setCommand(updatedTask.getCommand());

            // Ensure taskExecutions is initialized properly
            if (existingTask.getTaskExecutions() == null) {
                existingTask.setTaskExecutions(new ArrayList<>());
            }

            // Create new execution
            TaskExecution execution = new TaskExecution();
            execution.setStartTime(new Date());

            // Check for unsafe commands
            String output;
            if (isDangerousCommand(updatedTask.getCommand())) {
                output = "Unsafe command detected: " + updatedTask.getCommand();
            } else {
                output = executeShellCommand(updatedTask.getCommand());
                if (output == null || output.trim().isEmpty()) {
                    output = "Command executed successfully (No output)";
                }
            }

            execution.setOutput(output);
            execution.setEndTime(new Date());

            // Append the new execution to the existing list
            existingTask.getTaskExecutions().add(execution);

            // Save the task with updated executions
            return taskRepository.save(existingTask);
        } else {
            throw new RuntimeException("Task not found");
        }
    }



    @DeleteMapping("/{taskId}")
    public String deleteTask(@PathVariable String taskId) {
        if (taskRepository.existsById(taskId)) {
            taskRepository.deleteById(taskId);
            return "Task deleted successfully.";
        } else {
            throw new RuntimeException("Task not found");
        }
    }

    // Secure Shell Command Execution with 6s Timeout
    private String executeShellCommand(String command) {
        StringBuilder output = new StringBuilder();
        Process process = null;
        try {
            ProcessBuilder processBuilder = new ProcessBuilder("bash", "-c", command);
            processBuilder.redirectErrorStream(true);
            process = processBuilder.start();

            boolean finished = process.waitFor(5, TimeUnit.SECONDS);
            if (!finished) {
                process.destroy();
                return "Command execution timed out!";
            }

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append("\n");
                }
            }
        } catch (IOException | InterruptedException e) {
            return "Error executing command: " + e.getMessage();
        } finally {
            if (process != null) {
                process.destroy();
            }
        }

        if (output.toString().trim().isEmpty()) {
            return "Command executed successfully, no output!";
        }

        return output.toString().trim();
    }

    // Validate if the command is dangerous
    private boolean isDangerousCommand(String command) {
        return DANGEROUS_COMMANDS.stream().anyMatch(command.toLowerCase()::contains);
    }
}
