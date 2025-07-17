#output the database endpoint
output "db_endpoint" {
    value = aws_db_instance.db.endpoint
}